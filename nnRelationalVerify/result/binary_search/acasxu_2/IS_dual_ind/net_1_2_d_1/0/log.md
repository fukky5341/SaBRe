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
execution time: IAR + LP analysis = 1.24 + 1.99 = 3.22 seconds
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
rel_dist={0: [-1757.1236761183195, 1757.12367608548]}

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
rel_dist={0: [-1757.1235083509137, 1757.1235083132788]}

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
Binary search time: 65.74 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1131.03 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1240331, upper bound: 1757.1069627
time: 0.97 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1063020
time: 0.72 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.80 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.80
Output dim: 0, lower bound: -1757.1240331, upper bound: 1757.1069627
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.80
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1063020

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -356.5766602, 1445.0728760, -395.7757874, 1606.8325195, -1963.4089355, 1840.8486328
1: -437.0072937, 1613.3231201, -485.0866699, 1794.6639404, -2231.6711426, 2098.4096680
2: -500.5809937, 1640.3117676, -554.9915771, 1824.4812012, -2325.0620117, 2195.3032227
3: -710.3735962, 1789.6778564, -788.0418701, 1987.3795166, -2697.7529297, 2577.7197266
4: -841.4490967, 1668.5886230, -931.7708130, 1855.1873779, -2696.6364746, 2600.3593750

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1060141, upper bound: 1757.1060141
time: 0.59 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1060141, upper bound: 1757.1060569
time: 0.64 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -457.7469177, 1887.2629395, -392.1190796, 1592.8139648, -2050.5607910, 2279.3818359
1: -559.7200317, 2107.2399902, -480.5942993, 1778.9794922, -2338.6992188, 2587.8342285
2: -649.0897827, 2136.3076172, -549.9753418, 1808.4610596, -2457.5507812, 2686.2824707
3: -917.2413330, 2344.1276855, -780.8599243, 1969.9682617, -2887.2094727, 3124.9873047
4: -1100.8392334, 2173.5678711, -923.4982910, 1838.9149170, -2939.7541504, 3097.0661621

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1060569, upper bound: 1757.1062591
time: 0.66 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1060569, upper bound: 1757.1063020
time: 0.69 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.65 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.65
Output dim: 0, lower bound: -1757.1060141, upper bound: 1757.1060141
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.65
Output dim: 0, lower bound: -1757.1060141, upper bound: 1757.1060569
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.65
Output dim: 0, lower bound: -1757.1060569, upper bound: 1757.1062591
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.65
Output dim: 0, lower bound: -1757.1060569, upper bound: 1757.1063020

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -356.5766602, 1445.0728760, -356.5766602, 1445.0728760, -1801.6495361, 1801.6495361
1: -437.0072937, 1613.3231201, -437.0072937, 1613.3231201, -2050.3303223, 2050.3303223
2: -500.5809937, 1640.3117676, -500.5809937, 1640.3117676, -2140.8925781, 2140.8928223
3: -710.3735962, 1789.6778564, -710.3735962, 1789.6778564, -2500.0515137, 2500.0515137
4: -841.4490967, 1668.5886230, -841.4490967, 1668.5886230, -2510.0375977, 2510.0375977

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1238790, upper bound: 1757.1066803
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1230336, upper bound: 1757.1069199
time: 0.86 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -356.5766602, 1445.0728760, -457.7469177, 1887.2629395, -2243.8393555, 1902.8198242
1: -437.0072937, 1613.3231201, -559.7200317, 2107.2399902, -2544.2473145, 2173.0424805
2: -500.5809937, 1640.3117676, -649.0897827, 2136.3076172, -2636.8884277, 2289.4016113
3: -710.3735962, 1789.6778564, -917.2413330, 2344.1276855, -3054.5012207, 2706.9191895
4: -841.4490967, 1668.5886230, -1100.8392334, 2173.5678711, -3015.0170898, 2769.4277344

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1238790, upper bound: 1757.1067231
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1230336, upper bound: 1757.1069627
time: 0.70 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -457.7469177, 1887.2629395, -356.5766602, 1445.0728760, -1902.8198242, 2243.8393555
1: -559.7200317, 2107.2399902, -437.0072937, 1613.3231201, -2173.0427246, 2544.2473145
2: -649.0897827, 2136.3076172, -500.5809937, 1640.3117676, -2289.4016113, 2636.8886719
3: -917.2413330, 2344.1276855, -710.3735962, 1789.6778564, -2706.9191895, 3054.5012207
4: -1100.8392334, 2173.5678711, -841.4490967, 1668.5886230, -2769.4277344, 3015.0170898

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0404771, upper bound: 1757.1019499
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1060569, upper bound: 1757.1062591
time: 0.84 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -457.7469177, 1887.2629395, -457.7469177, 1887.2629395, -2345.0095215, 2345.0095215
1: -559.7200317, 2107.2399902, -559.7200317, 2107.2399902, -2666.9597168, 2666.9597168
2: -649.0897827, 2136.3076172, -649.0897827, 2136.3076172, -2785.3974609, 2785.3974609
3: -917.2413330, 2344.1276855, -917.2413330, 2344.1276855, -3261.3691406, 3261.3691406
4: -1100.8392334, 2173.5678711, -1100.8392334, 2173.5678711, -3274.4072266, 3274.4072266

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0404776, upper bound: 1757.1019499
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1060564, upper bound: 1757.1063020
time: 0.70 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.69 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -1757.1238790, upper bound: 1757.1066803
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -1757.1230336, upper bound: 1757.1069199
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -1757.1238790, upper bound: 1757.1067231
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -1757.1230336, upper bound: 1757.1069627
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -1757.0404771, upper bound: 1757.1019499
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -1757.1060569, upper bound: 1757.1062591
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -1757.0404776, upper bound: 1757.1019499
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -1757.1060564, upper bound: 1757.1063020

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -314.7524109, 1277.9980469, -356.0819702, 1443.0485840, -1757.8009033, 1634.0798340
1: -385.6337891, 1426.8695068, -436.4061279, 1611.0620117, -1996.6954346, 1863.2756348
2: -441.0567932, 1450.1931152, -499.8819275, 1638.0190430, -2079.0756836, 1950.0750732
3: -626.3041992, 1580.1624756, -709.3872681, 1787.1778564, -2413.4819336, 2289.5498047
4: -740.3444824, 1474.6290283, -840.2650146, 1666.2618408, -2406.6064453, 2314.8940430

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1236998, upper bound: 1757.1236998
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1236998, upper bound: 1757.1236998
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -352.7528076, 1429.8717041, -356.5766602, 1445.0728760, -1797.8256836, 1786.4483643
1: -432.3590393, 1596.3819580, -437.0072937, 1613.3231201, -2045.6821289, 2033.3892822
2: -495.2670288, 1623.0344238, -500.5809937, 1640.3117676, -2135.5786133, 2123.6154785
3: -702.8384399, 1770.8116455, -710.3735962, 1789.6778564, -2492.5161133, 2481.1850586
4: -832.5136719, 1651.0700684, -841.4490967, 1668.5886230, -2501.1022949, 2492.5190430

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1236998, upper bound: 1757.1239394
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1236998, upper bound: 1757.1239394
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -314.7524109, 1277.9980469, -457.3528748, 1885.6625977, -2200.4150391, 1735.3509521
1: -385.6337891, 1426.8695068, -559.2342529, 2105.4519043, -2491.0856934, 1986.1037598
2: -441.0567932, 1450.1931152, -648.5374146, 2134.4899902, -2575.5468750, 2098.7304688
3: -626.3041992, 1580.1624756, -916.4539185, 2342.1623535, -2968.4665527, 2496.6162109
4: -740.3444824, 1474.6290283, -1099.9218750, 2171.7268066, -2912.0712891, 2574.5505371

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1189694, upper bound: 1757.0411438
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1189694, upper bound: 1757.1067231
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -352.7528076, 1429.8717041, -457.7469177, 1887.2629395, -2240.0156250, 1887.6186523
1: -432.3590393, 1596.3819580, -559.7200317, 2107.2399902, -2539.5988770, 2156.1018066
2: -495.2670288, 1623.0344238, -649.0897827, 2136.3076172, -2631.5744629, 2272.1242676
3: -702.8384399, 1770.8116455, -917.2413330, 2344.1276855, -3046.9660645, 2688.0529785
4: -832.5136719, 1651.0700684, -1100.8392334, 2173.5678711, -3006.0815430, 2751.9091797

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1189694, upper bound: 1757.0413833
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1189694, upper bound: 1757.1069627
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -393.3483887, 1617.7186279, -356.0819702, 1443.0485840, -1836.3969727, 1973.8004150
1: -481.2612000, 1806.1429443, -436.4061279, 1611.0620117, -2092.3229980, 2242.5490723
2: -557.5192261, 1832.2569580, -499.8819275, 1638.0190430, -2195.5380859, 2332.1386719
3: -787.0695801, 2007.7425537, -709.3872681, 1787.1778564, -2574.2475586, 2717.1298828
4: -943.0974121, 1864.2763672, -840.2650146, 1666.2618408, -2609.3588867, 2704.5415039

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0411438, upper bound: 1757.1189694
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0411438, upper bound: 1757.1189694
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -450.1930237, 1856.5698242, -356.5766602, 1445.0728760, -1895.2657471, 2213.1462402
1: -550.5508423, 2073.0170898, -437.0072937, 1613.3231201, -2163.8735352, 2510.0241699
2: -638.3911133, 2101.5507812, -500.5809937, 1640.3117676, -2278.7028809, 2602.1318359
3: -902.1766968, 2305.8959961, -710.3735962, 1789.6778564, -2691.8544922, 3016.2695312
4: -1082.5723877, 2138.3459473, -841.4490967, 1668.5886230, -2751.1606445, 2979.7949219

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1067231, upper bound: 1757.1232787
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1067231, upper bound: 1757.1232787
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -393.3483887, 1617.7186279, -457.3528748, 1885.6625977, -2279.0109863, 2075.0715332
1: -481.2612000, 1806.1429443, -559.2342529, 2105.4519043, -2586.7131348, 2365.3769531
2: -557.5192261, 1832.2569580, -648.5374146, 2134.4899902, -2692.0092773, 2480.7944336
3: -787.0695801, 2007.7425537, -916.4539185, 2342.1623535, -3129.2319336, 2924.1960449
4: -943.0974121, 1864.2763672, -1099.9218750, 2171.7268066, -3114.8239746, 2964.1982422

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0364133, upper bound: 1757.0364133
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0364134, upper bound: 1757.1019499
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -450.1930237, 1856.5698242, -457.7469177, 1887.2629395, -2337.4558105, 2314.3166504
1: -550.5508423, 2073.0170898, -559.7200317, 2107.2399902, -2657.7907715, 2632.7363281
2: -638.3911133, 2101.5507812, -649.0897827, 2136.3076172, -2774.6987305, 2750.6406250
3: -902.1766968, 2305.8959961, -917.2413330, 2344.1276855, -3246.3044434, 3223.1372070
4: -1082.5723877, 2138.3459473, -1100.8392334, 2173.5678711, -3256.1396484, 3239.1850586

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1019926, upper bound: 1757.0407226
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1019926, upper bound: 1757.1063020
time: 0.68 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.68 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -1757.1236998, upper bound: 1757.1236998
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -1757.1236998, upper bound: 1757.1236998
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -1757.1236998, upper bound: 1757.1239394
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -1757.1236998, upper bound: 1757.1239394
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -1757.1189694, upper bound: 1757.0411438
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -1757.1189694, upper bound: 1757.1067231
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -1757.1189694, upper bound: 1757.0413833
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -1757.1189694, upper bound: 1757.1069627
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -1757.0411438, upper bound: 1757.1189694
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -1757.0411438, upper bound: 1757.1189694
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -1757.1067231, upper bound: 1757.1232787
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -1757.1067231, upper bound: 1757.1232787
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -1757.0364133, upper bound: 1757.0364133
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -1757.0364134, upper bound: 1757.1019499
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -1757.1019926, upper bound: 1757.0407226
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -1757.1019926, upper bound: 1757.1063020

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -314.7524109, 1277.9980469, -314.7524109, 1277.9980469, -1592.7503662, 1592.7503662
1: -385.6337891, 1426.8695068, -385.6337891, 1426.8695068, -1812.5030518, 1812.5030518
2: -441.0567932, 1450.1931152, -441.0567932, 1450.1931152, -1891.2498779, 1891.2498779
3: -626.3041992, 1580.1624756, -626.3041992, 1580.1624756, -2206.4663086, 2206.4663086
4: -740.3444824, 1474.6290283, -740.3444824, 1474.6290283, -2214.9733887, 2214.9731445

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1181179, upper bound: 1757.1116917
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1152449, upper bound: 1757.1115415
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -314.7524109, 1277.9980469, -352.7528076, 1429.8717041, -1744.6240234, 1630.7508545
1: -385.6337891, 1426.8695068, -432.3590393, 1596.3819580, -1982.0155029, 1859.2283936
2: -441.0567932, 1450.1931152, -495.2670288, 1623.0344238, -2064.0913086, 1945.4600830
3: -626.3041992, 1580.1624756, -702.8384399, 1770.8116455, -2397.1157227, 2283.0007324
4: -740.3444824, 1474.6290283, -832.5136719, 1651.0700684, -2391.4145508, 2307.1425781

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1181179, upper bound: 1757.1116917
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1152449, upper bound: 1757.1115415
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -352.7528076, 1429.8717041, -314.7524109, 1277.9980469, -1630.7508545, 1744.6241455
1: -432.3590393, 1596.3819580, -385.6337891, 1426.8695068, -1859.2283936, 1982.0156250
2: -495.2670288, 1623.0344238, -441.0567932, 1450.1931152, -1945.4600830, 2064.0913086
3: -702.8384399, 1770.8116455, -626.3041992, 1580.1624756, -2283.0007324, 2397.1157227
4: -832.5136719, 1651.0700684, -740.3444824, 1474.6290283, -2307.1425781, 2391.4145508

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1182722, upper bound: 1757.1119313
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1115415, upper bound: 1757.1116194
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -352.7528076, 1429.8717041, -352.7528076, 1429.8717041, -1782.6245117, 1782.6245117
1: -432.3590393, 1596.3819580, -432.3590393, 1596.3819580, -2028.7409668, 2028.7409668
2: -495.2670288, 1623.0344238, -495.2670288, 1623.0344238, -2118.3015137, 2118.3015137
3: -702.8384399, 1770.8116455, -702.8384399, 1770.8116455, -2473.6499023, 2473.6499023
4: -832.5136719, 1651.0700684, -832.5136719, 1651.0700684, -2483.5837402, 2483.5837402

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1182722, upper bound: 1757.1119313
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1115415, upper bound: 1757.1116194
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -314.7524109, 1277.9980469, -393.3483887, 1617.7186279, -1932.4709473, 1671.3464355
1: -385.6337891, 1426.8695068, -481.2612000, 1806.1429443, -2191.7768555, 1908.1303711
2: -441.0567932, 1450.1931152, -557.5192261, 1832.2569580, -2273.3137207, 2007.7124023
3: -626.3041992, 1580.1624756, -787.0695801, 2007.7425537, -2634.0463867, 2367.2319336
4: -740.3444824, 1474.6290283, -943.0974121, 1864.2763672, -2604.6208496, 2417.7255859

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1135376, upper bound: 1757.0411437
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1106646, upper bound: 1757.0409936
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -314.7524109, 1277.9980469, -450.1930237, 1856.5698242, -2171.3222656, 1728.1907959
1: -385.6337891, 1426.8695068, -550.5508423, 2073.0170898, -2458.6503906, 1977.4204102
2: -441.0567932, 1450.1931152, -638.3911133, 2101.5507812, -2542.6076660, 2088.5842285
3: -626.3041992, 1580.1624756, -902.1766968, 2305.8959961, -2932.2001953, 2482.3388672
4: -740.3444824, 1474.6290283, -1082.5723877, 2138.3459473, -2878.6904297, 2557.2004395

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1135376, upper bound: 1757.0958569
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1106646, upper bound: 1757.0957068
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -352.7528076, 1429.8717041, -393.3483887, 1617.7186279, -1970.4714355, 1823.2200928
1: -432.3590393, 1596.3819580, -481.2612000, 1806.1429443, -2238.5019531, 2077.6430664
2: -495.2670288, 1623.0344238, -557.5192261, 1832.2569580, -2327.5239258, 2180.5537109
3: -702.8384399, 1770.8116455, -787.0695801, 2007.7425537, -2710.5805664, 2557.8813477
4: -832.5136719, 1651.0700684, -943.0974121, 1864.2763672, -2696.7900391, 2594.1669922

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1136919, upper bound: 1757.0413833
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1069613, upper bound: 1757.0410715
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -352.7528076, 1429.8717041, -450.1930237, 1856.5698242, -2209.3227539, 1880.0646973
1: -432.3590393, 1596.3819580, -550.5508423, 2073.0170898, -2505.3759766, 2146.9328613
2: -495.2670288, 1623.0344238, -638.3911133, 2101.5507812, -2596.8178711, 2261.4255371
3: -702.8384399, 1770.8116455, -902.1766968, 2305.8959961, -3008.7343750, 2672.9882812
4: -832.5136719, 1651.0700684, -1082.5723877, 2138.3459473, -2970.8596191, 2733.6418457

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1136919, upper bound: 1757.0954967
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1069612, upper bound: 1757.0951746
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -393.3483887, 1617.7186279, -314.7524109, 1277.9980469, -1671.3464355, 1932.4709473
1: -481.2612000, 1806.1429443, -385.6337891, 1426.8695068, -1908.1303711, 2191.7768555
2: -557.5192261, 1832.2569580, -441.0567932, 1450.1931152, -2007.7124023, 2273.3137207
3: -787.0695801, 2007.7425537, -626.3041992, 1580.1624756, -2367.2319336, 2634.0463867
4: -943.0974121, 1864.2763672, -740.3444824, 1474.6290283, -2417.7255859, 2604.6208496

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0335044, upper bound: 1757.1189659
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0361626, upper bound: 1757.0366233
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0349349, upper bound: 1757.0363111
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -393.3483887, 1617.7186279, -352.7528076, 1429.8717041, -1823.2200928, 1970.4714355
1: -481.2612000, 1806.1429443, -432.3590393, 1596.3819580, -2077.6430664, 2238.5019531
2: -557.5192261, 1832.2569580, -495.2670288, 1623.0344238, -2180.5537109, 2327.5239258
3: -787.0695801, 2007.7425537, -702.8384399, 1770.8116455, -2557.8813477, 2710.5805664
4: -943.0974121, 1864.2763672, -832.5136719, 1651.0700684, -2594.1669922, 2696.7900391

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0335044, upper bound: 1757.1189659
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0361626, upper bound: 1757.1153521
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0349349, upper bound: 1757.1150400
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -450.1930237, 1856.5698242, -314.7524109, 1277.9980469, -1728.1907959, 2171.3220215
1: -550.5508423, 2073.0170898, -385.6337891, 1426.8695068, -1977.4204102, 2458.6503906
2: -638.3911133, 2101.5507812, -441.0567932, 1450.1931152, -2088.5842285, 2542.6076660
3: -902.1766968, 2305.8959961, -626.3041992, 1580.1624756, -2482.3388672, 2932.2001953
4: -1082.5723877, 2138.3459473, -740.3444824, 1474.6290283, -2557.2004395, 2878.6904297

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0977432, upper bound: 1757.1112705
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0957068, upper bound: 1757.1112705
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -450.1930237, 1856.5698242, -352.7528076, 1429.8717041, -1880.0645752, 2209.3227539
1: -550.5508423, 2073.0170898, -432.3590393, 1596.3819580, -2146.9328613, 2505.3757324
2: -638.3911133, 2101.5507812, -495.2670288, 1623.0344238, -2261.4255371, 2596.8178711
3: -902.1766968, 2305.8959961, -702.8384399, 1770.8116455, -2672.9882812, 3008.7343750
4: -1082.5723877, 2138.3459473, -832.5136719, 1651.0700684, -2733.6418457, 2970.8596191

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0977432, upper bound: 1757.1112705
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0957068, upper bound: 1757.1112705
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -393.3483887, 1617.7186279, -450.1930237, 1856.5698242, -2249.9177246, 2067.9116211
1: -481.2612000, 1806.1429443, -550.5508423, 2073.0170898, -2554.2778320, 2356.6938477
2: -557.5192261, 1832.2569580, -638.3911133, 2101.5507812, -2659.0700684, 2470.6479492
3: -787.0695801, 2007.7425537, -902.1766968, 2305.8959961, -3092.9655762, 2909.9189453
4: -943.0974121, 1864.2763672, -1082.5723877, 2138.3459473, -3081.4428711, 2946.8481445

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0361011, upper bound: 1757.1019499
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0348734, upper bound: 1757.1016639
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -450.1930237, 1856.5698242, -393.3483887, 1617.7186279, -2067.9116211, 2249.9179688
1: -550.5508423, 2073.0170898, -481.2612000, 1806.1429443, -2356.6936035, 2554.2780762
2: -638.3911133, 2101.5507812, -557.5192261, 1832.2569580, -2470.6479492, 2659.0700684
3: -902.1766968, 2305.8959961, -787.0695801, 2007.7425537, -2909.9189453, 3092.9655762
4: -1082.5723877, 2138.3459473, -943.0974121, 1864.2763672, -2946.8481445, 3081.4428711

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0931630, upper bound: 1757.0407226
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0911265, upper bound: 1757.0407226
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -450.1930237, 1856.5698242, -450.1930237, 1856.5698242, -2306.7626953, 2306.7626953
1: -550.5508423, 2073.0170898, -550.5508423, 2073.0170898, -2623.5678711, 2623.5678711
2: -638.3911133, 2101.5507812, -638.3911133, 2101.5507812, -2739.9418945, 2739.9418945
3: -902.1766968, 2305.8959961, -902.1766968, 2305.8959961, -3208.0727539, 3208.0727539
4: -1082.5723877, 2138.3459473, -1082.5723877, 2138.3459473, -3220.9177246, 3220.9177246

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0931630, upper bound: 1757.0954358
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0911265, upper bound: 1757.0954358
time: 0.69 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.85 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -1757.1181179, upper bound: 1757.1116917
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -1757.1152449, upper bound: 1757.1115415
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -1757.1181179, upper bound: 1757.1116917
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -1757.1152449, upper bound: 1757.1115415
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -1757.1182722, upper bound: 1757.1119313
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -1757.1115415, upper bound: 1757.1116194
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -1757.1182722, upper bound: 1757.1119313
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -1757.1115415, upper bound: 1757.1116194
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -1757.1135376, upper bound: 1757.0411437
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -1757.1106646, upper bound: 1757.0409936
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -1757.1135376, upper bound: 1757.0958569
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -1757.1106646, upper bound: 1757.0957068
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -1757.1136919, upper bound: 1757.0413833
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -1757.1069613, upper bound: 1757.0410715
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -1757.1136919, upper bound: 1757.0954967
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -1757.1069612, upper bound: 1757.0951746
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.85
Output dim: 0, lower bound: -1757.0361626, upper bound: 1757.0366233
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.85
Output dim: 0, lower bound: -1757.0349349, upper bound: 1757.0363111
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -1757.0361626, upper bound: 1757.1153521
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -1757.0349349, upper bound: 1757.1150400
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -1757.0977432, upper bound: 1757.1112705
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -1757.0957068, upper bound: 1757.1112705
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -1757.0977432, upper bound: 1757.1112705
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -1757.0957068, upper bound: 1757.1112705
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -1757.0361011, upper bound: 1757.1019499
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -1757.0348734, upper bound: 1757.1016639
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -1757.0931630, upper bound: 1757.0407226
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -1757.0911265, upper bound: 1757.0407226
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -1757.0931630, upper bound: 1757.0954358
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -1757.0911265, upper bound: 1757.0954358

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -272.6042480, 1111.4747314, -314.2249146, 1275.8792725, -1548.4835205, 1425.6993408
1: -334.2237854, 1241.4110107, -384.9903564, 1424.5043945, -1758.7281494, 1626.4013672
2: -381.0059814, 1261.2246094, -440.3281555, 1447.7878418, -1828.7937012, 1701.5527344
3: -542.2633057, 1371.0507812, -625.2636719, 1577.5506592, -2119.8132324, 1996.3144531
4: -637.6555786, 1281.7025146, -739.1367188, 1472.1783447, -2109.8339844, 2020.8392334

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1148372, upper bound: 1757.1148372
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1148372, upper bound: 1757.1152449
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -311.7589417, 1265.6518555, -314.7524109, 1277.9980469, -1589.7569580, 1580.4041748
1: -381.9956665, 1413.0462646, -385.6337891, 1426.8695068, -1808.8651123, 1798.6798096
2: -436.8746033, 1436.2211914, -441.0567932, 1450.1931152, -1887.0676270, 1877.2778320
3: -620.3742065, 1564.9211426, -626.3041992, 1580.1624756, -2200.5366211, 2191.2250977
4: -733.2927246, 1460.4571533, -740.3444824, 1474.6290283, -2207.9216309, 2200.8015137

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1152449, upper bound: 1757.1148372
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1152449, upper bound: 1757.1152449
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -272.6042480, 1111.4747314, -352.1714172, 1427.5303955, -1700.1346436, 1463.6459961
1: -334.2237854, 1241.4110107, -431.6478577, 1593.7680664, -1927.9916992, 1673.0588379
2: -381.0059814, 1261.2246094, -494.4586487, 1620.3793945, -2001.3852539, 1755.6832275
3: -542.2633057, 1371.0507812, -701.6849976, 1767.9172363, -2310.1801758, 2072.7355957
4: -637.6555786, 1281.7025146, -831.1648560, 1648.3659668, -2286.0214844, 2112.8674316

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1149145, upper bound: 1757.1115415
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1149145, upper bound: 1757.1115415
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -311.7589417, 1265.6518555, -352.7528076, 1429.8717041, -1741.6306152, 1618.4046631
1: -381.9956665, 1413.0462646, -432.3590393, 1596.3819580, -1978.3775635, 1845.4051514
2: -436.8746033, 1436.2211914, -495.2670288, 1623.0344238, -2059.9089355, 1931.4881592
3: -620.3742065, 1564.9211426, -702.8384399, 1770.8116455, -2391.1855469, 2267.7595215
4: -733.2927246, 1460.4571533, -832.5136719, 1651.0700684, -2384.3627930, 2292.9707031

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1152449, upper bound: 1757.1115415
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1153216, upper bound: 1757.1115415
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -309.5700378, 1259.4353027, -314.2249146, 1275.8792725, -1585.4490967, 1573.6599121
1: -379.7421265, 1406.8647461, -384.9903564, 1424.5043945, -1804.2465820, 1791.8551025
2: -433.3220215, 1429.6942139, -440.3281555, 1447.7878418, -1881.1098633, 1870.0222168
3: -616.7457275, 1556.0572510, -625.2636719, 1577.5506592, -2194.2961426, 2181.3208008
4: -726.5832520, 1453.5422363, -739.1367188, 1472.1783447, -2198.7614746, 2192.6789551

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1115415, upper bound: 1757.1149145
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1115415, upper bound: 1757.1153216
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -349.7909546, 1417.5225830, -314.7524109, 1277.9980469, -1627.7889404, 1732.2750244
1: -428.7479553, 1582.5559082, -385.6337891, 1426.8695068, -1855.6174316, 1968.1894531
2: -491.1026001, 1609.0854492, -441.0567932, 1450.1931152, -1941.2956543, 2050.1423340
3: -696.9265137, 1755.5729980, -626.3041992, 1580.1624756, -2277.0886230, 2381.8769531
4: -825.4575806, 1636.9227295, -740.3444824, 1474.6290283, -2300.0861816, 2377.2668457

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1115415, upper bound: 1757.1149145
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1115415, upper bound: 1757.1153216
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -309.5700378, 1259.4353027, -352.1714172, 1427.5303955, -1737.1004639, 1611.6066895
1: -379.7421265, 1406.8647461, -431.6478577, 1593.7680664, -1973.5102539, 1838.5125732
2: -433.3220215, 1429.6942139, -494.4586487, 1620.3793945, -2053.7014160, 1924.1528320
3: -616.7457275, 1556.0572510, -701.6849976, 1767.9172363, -2384.6630859, 2257.7421875
4: -726.5832520, 1453.5422363, -831.1648560, 1648.3659668, -2374.9489746, 2284.7070312

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1115415, upper bound: 1757.1116194
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1115415, upper bound: 1757.1116194
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -349.7909546, 1417.5225830, -352.7528076, 1429.8717041, -1779.6625977, 1770.2753906
1: -428.7479553, 1582.5559082, -432.3590393, 1596.3819580, -2025.1298828, 2014.9147949
2: -491.1026001, 1609.0854492, -495.2670288, 1623.0344238, -2114.1369629, 2104.3525391
3: -696.9265137, 1755.5729980, -702.8384399, 1770.8116455, -2467.7380371, 2458.4113770
4: -825.4575806, 1636.9227295, -832.5136719, 1651.0700684, -2476.5275879, 2469.4362793

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1115415, upper bound: 1757.1116194
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1115415, upper bound: 1757.1116194
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -272.6042480, 1111.4747314, -393.0056458, 1616.3446045, -1888.9488525, 1504.4801025
1: -334.2237854, 1241.4110107, -480.8356018, 1804.6072998, -2138.8305664, 1722.2465820
2: -381.0059814, 1261.2246094, -557.0473633, 1830.6922607, -2211.6979980, 1818.2719727
3: -542.2633057, 1371.0507812, -786.3866577, 2006.0635986, -2548.3261719, 2157.4375000
4: -637.6555786, 1281.7025146, -942.3229370, 1862.6791992, -2500.3347168, 2224.0253906

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1135341, upper bound: 1757.0335044
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1112795, upper bound: 1757.0276145
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1134642, upper bound: 1757.0411438
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1135376, upper bound: 1757.0411437
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1118868, upper bound: 1757.0407673
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1114347, upper bound: 1757.0409177
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1135376, upper bound: 1757.0411437
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1125668, upper bound: 1757.0308572
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -311.7589417, 1265.6518555, -393.3483887, 1617.7186279, -1929.4775391, 1659.0002441
1: -381.9956665, 1413.0462646, -481.2612000, 1806.1429443, -2188.1381836, 1894.3071289
2: -436.8746033, 1436.2211914, -557.5192261, 1832.2569580, -2269.1315918, 1993.7404785
3: -620.3742065, 1564.9211426, -787.0695801, 2007.7425537, -2628.1162109, 2351.9907227
4: -733.2927246, 1460.4571533, -943.0974121, 1864.2763672, -2597.5690918, 2403.5541992

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1106610, upper bound: 1757.0333542
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0366233, upper bound: 1757.0361626
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0363111, upper bound: 1757.0349349
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -272.6042480, 1111.4747314, -449.8562927, 1855.2331543, -2127.8374023, 1561.3309326
1: -334.2237854, 1241.4110107, -550.1342773, 2071.5241699, -2405.7475586, 1791.5451660
2: -381.0059814, 1261.2246094, -637.9258423, 2100.0305176, -2481.0356445, 1899.1503906
3: -542.2633057, 1371.0507812, -901.5075684, 2304.2578125, -2846.5209961, 2272.5583496
4: -637.6555786, 1281.7025146, -1081.8038330, 2136.7978516, -2774.4533691, 2363.5063477

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1144086, upper bound: 1757.0957068
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1144086, upper bound: 1757.0957068
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -311.7589417, 1265.6518555, -450.1930237, 1856.5698242, -2168.3283691, 1715.8446045
1: -381.9956665, 1413.0462646, -550.5508423, 2073.0170898, -2455.0119629, 1963.5971680
2: -436.8746033, 1436.2211914, -638.3911133, 2101.5507812, -2538.4252930, 2074.6123047
3: -620.3742065, 1564.9211426, -902.1766968, 2305.8959961, -2926.2700195, 2467.0979004
4: -733.2927246, 1460.4571533, -1082.5723877, 2138.3459473, -2871.6386719, 2543.0290527

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1144086, upper bound: 1757.0957068
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1144086, upper bound: 1757.0957068
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -309.5700378, 1259.4353027, -393.0056458, 1616.3446045, -1925.9145508, 1652.4406738
1: -379.7421265, 1406.8647461, -480.8356018, 1804.6072998, -2184.3493652, 1887.7003174
2: -433.3220215, 1429.6942139, -557.0473633, 1830.6922607, -2264.0141602, 1986.7415771
3: -616.7457275, 1556.0572510, -786.3866577, 2006.0635986, -2622.8090820, 2342.4438477
4: -726.5832520, 1453.5422363, -942.3229370, 1862.6791992, -2589.2624512, 2395.8649902

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1091557, upper bound: 1757.0409484
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1136884, upper bound: 1757.0337440
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1070679, upper bound: 1757.0406799
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1067557, upper bound: 1757.0394521
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -349.7909546, 1417.5225830, -393.3483887, 1617.7186279, -1967.5095215, 1810.8709717
1: -428.7479553, 1582.5559082, -481.2612000, 1806.1429443, -2234.8908691, 2063.8171387
2: -491.1026001, 1609.0854492, -557.5192261, 1832.2569580, -2323.3591309, 2166.6047363
3: -696.9265137, 1755.5729980, -787.0695801, 2007.7425537, -2704.6684570, 2542.6425781
4: -825.4575806, 1636.9227295, -943.0974121, 1864.2763672, -2689.7338867, 2580.0192871

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0988347, upper bound: 1757.0406481
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1069577, upper bound: 1757.0334321
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1035587, upper bound: 1757.0405908
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1032465, upper bound: 1757.0393631
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -309.5700378, 1259.4353027, -449.8562927, 1855.2331543, -2164.8032227, 1709.2915039
1: -379.7421265, 1406.8647461, -550.1342773, 2071.5241699, -2451.2661133, 1956.9990234
2: -433.3220215, 1429.6942139, -637.9258423, 2100.0305176, -2533.3525391, 2067.6201172
3: -616.7457275, 1556.0572510, -901.5075684, 2304.2578125, -2921.0034180, 2457.5649414
4: -726.5832520, 1453.5422363, -1081.8038330, 2136.7978516, -2863.3806152, 2535.3459473

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0953112, upper bound: 1757.0951746
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0953112, upper bound: 1757.0951746
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -349.7909546, 1417.5225830, -450.1930237, 1856.5698242, -2206.3605957, 1867.7155762
1: -428.7479553, 1582.5559082, -550.5508423, 2073.0170898, -2501.7651367, 2133.1066895
2: -491.1026001, 1609.0854492, -638.3911133, 2101.5507812, -2592.6530762, 2247.4765625
3: -696.9265137, 1755.5729980, -902.1766968, 2305.8959961, -3002.8225098, 2657.7497559
4: -825.4575806, 1636.9227295, -1082.5723877, 2138.3459473, -2963.8034668, 2719.4941406

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0953112, upper bound: 1757.0951746
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0953112, upper bound: 1757.0951746
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -378.1842957, 1554.5784912, -352.7528076, 1429.8717041, -1808.0560303, 1907.3312988
1: -462.3816833, 1735.4530029, -432.3590393, 1596.3819580, -2058.7636719, 2167.8115234
2: -536.3656006, 1760.1306152, -495.2670288, 1623.0344238, -2159.3999023, 2255.3977051
3: -756.5307007, 1930.6671143, -702.8384399, 1770.8116455, -2527.3422852, 2633.5046387
4: -908.3130493, 1790.7239990, -832.5136719, 1651.0700684, -2559.3830566, 2623.2377930

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0406799, upper bound: 1757.1070679
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0405908, upper bound: 1757.1035587
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -385.2216797, 1589.5644531, -352.0792236, 1427.1434326, -1812.3648682, 1941.6436768
1: -470.5504150, 1774.0218506, -431.5322266, 1593.3321533, -2063.8818359, 2205.5541992
2: -547.8798828, 1799.2845459, -494.3213806, 1619.9329834, -2167.8127441, 2293.6059570
3: -771.5361328, 1976.5567627, -701.4971924, 1767.4089355, -2538.9450684, 2678.0539551
4: -930.1417847, 1831.4566650, -830.9238281, 1647.9057617, -2578.0476074, 2662.3803711

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0394521, upper bound: 1757.1067557
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0393631, upper bound: 1757.1032465
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -418.2335815, 1726.7127686, -314.2249146, 1275.8792725, -1694.1125488, 2040.9372559
1: -511.6340027, 1928.8596191, -384.9903564, 1424.5043945, -1936.1383057, 2313.8500977
2: -592.1152344, 1955.3981934, -440.3281555, 1447.7878418, -2039.9028320, 2395.7263184
3: -837.6755371, 2141.4594727, -625.2636719, 1577.5506592, -2415.2260742, 2766.7231445
4: -1002.3061523, 1988.3162842, -739.1367188, 1472.1783447, -2474.4843750, 2727.4531250

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0957068, upper bound: 1757.1144086
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0957068, upper bound: 1757.1144086
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -443.2770081, 1826.9315186, -314.7524109, 1277.9980469, -1721.2750244, 2141.6838379
1: -542.2296143, 2039.9180908, -385.6337891, 1426.8695068, -1969.0991211, 2425.5512695
2: -628.5104370, 2068.2482910, -441.0567932, 1450.1931152, -2078.7036133, 2509.3051758
3: -888.3466797, 2269.0156250, -626.3041992, 1580.1624756, -2468.5087891, 2895.3195801
4: -1065.5307617, 2104.4018555, -740.3444824, 1474.6290283, -2540.1594238, 2844.7460938

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0957068, upper bound: 1757.1144086
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0957068, upper bound: 1757.1144086
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -418.2335815, 1726.7127686, -352.1714172, 1427.5303955, -1845.7639160, 2078.8840332
1: -511.6340027, 1928.8596191, -431.6478577, 1593.7680664, -2105.4020996, 2360.5075684
2: -592.1152344, 1955.3981934, -494.4586487, 1620.3793945, -2212.4946289, 2449.8564453
3: -837.6755371, 2141.4594727, -701.6849976, 1767.9172363, -2605.5927734, 2843.1445312
4: -1002.3061523, 1988.3162842, -831.1648560, 1648.3659668, -2650.6721191, 2819.4812012

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0957847, upper bound: 1757.1112705
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0957847, upper bound: 1757.1112705
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -443.2770081, 1826.9315186, -352.7528076, 1429.8717041, -1873.1486816, 2179.6843262
1: -542.2296143, 2039.9180908, -432.3590393, 1596.3819580, -2138.6115723, 2472.2766113
2: -628.5104370, 2068.2482910, -495.2670288, 1623.0344238, -2251.5449219, 2563.5151367
3: -888.3466797, 2269.0156250, -702.8384399, 1770.8116455, -2659.1582031, 2971.8540039
4: -1065.5307617, 2104.4018555, -832.5136719, 1651.0700684, -2716.6008301, 2936.9152832

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0957847, upper bound: 1757.1112705
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0957847, upper bound: 1757.1112705
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -378.1842957, 1554.5784912, -450.1930237, 1856.5698242, -2234.7541504, 2004.7713623
1: -462.3816833, 1735.4530029, -550.5508423, 2073.0170898, -2535.3986816, 2286.0036621
2: -536.3656006, 1760.1306152, -638.3911133, 2101.5507812, -2637.9165039, 2398.5217285
3: -756.5307007, 1930.6671143, -902.1766968, 2305.8959961, -3062.4267578, 2832.8430176
4: -908.3130493, 1790.7239990, -1082.5723877, 2138.3459473, -3046.6589355, 2873.2963867

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0402167, upper bound: 1757.0931629
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0402167, upper bound: 1757.0911106
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -385.2216797, 1589.5644531, -449.3044128, 1852.9316406, -2238.1533203, 2038.8687744
1: -470.5504150, 1774.0218506, -549.4618530, 2068.9570312, -2539.5073242, 2323.4836426
2: -547.8798828, 1799.2845459, -637.1530151, 2097.4189453, -2645.2988281, 2436.4375000
3: -771.5361328, 1976.5567627, -900.4041748, 2301.4047852, -3072.9409180, 2876.9604492
4: -930.1417847, 1831.4566650, -1080.4837646, 2134.1489258, -3064.2902832, 2911.9404297

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0389890, upper bound: 1757.0928507
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0389890, upper bound: 1757.0907983
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -418.2335815, 1726.7127686, -393.0056458, 1616.3446045, -2034.5780029, 2119.7182617
1: -511.6340027, 1928.8596191, -480.8356018, 1804.6072998, -2316.2412109, 2409.6953125
2: -592.1152344, 1955.3981934, -557.0473633, 1830.6922607, -2422.8071289, 2512.4453125
3: -837.6755371, 2141.4594727, -786.3866577, 2006.0635986, -2843.7387695, 2927.8461914
4: -1002.3061523, 1988.3162842, -942.3229370, 1862.6791992, -2864.9853516, 2930.6391602

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0927727, upper bound: 1757.0406486
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0931594, upper bound: 1757.0330832
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0931630, upper bound: 1757.0403857
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0928507, upper bound: 1757.0391580
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -443.2770081, 1826.9315186, -393.3483887, 1617.7186279, -2060.9956055, 2220.2795410
1: -542.2296143, 2039.9180908, -481.2612000, 1806.1429443, -2348.3725586, 2521.1787109
2: -628.5104370, 2068.2482910, -557.5192261, 1832.2569580, -2460.7673340, 2625.7673340
3: -888.3466797, 2269.0156250, -787.0695801, 2007.7425537, -2896.0886230, 3056.0852051
4: -1065.5307617, 2104.4018555, -943.0974121, 1864.2763672, -2929.8071289, 3047.4985352

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0907363, upper bound: 1757.0406486
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0911151, upper bound: 1757.0330832
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0911265, upper bound: 1757.0403857
time: 1.07 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0908144, upper bound: 1757.0391580
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -418.2335815, 1726.7127686, -449.8562927, 1855.2331543, -2273.4667969, 2176.5688477
1: -511.6340027, 1928.8596191, -550.1342773, 2071.5241699, -2583.1579590, 2478.9938965
2: -592.1152344, 1955.3981934, -637.9258423, 2100.0305176, -2692.1450195, 2593.3239746
3: -837.6755371, 2141.4594727, -901.5075684, 2304.2578125, -3141.9333496, 3042.9670410
4: -1002.3061523, 1988.3162842, -1081.8038330, 2136.7978516, -3139.1037598, 3070.1201172

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0953052, upper bound: 1757.0954358
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0953052, upper bound: 1757.0954358
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -443.2770081, 1826.9315186, -450.1930237, 1856.5698242, -2299.8461914, 2277.1242676
1: -542.2296143, 2039.9180908, -550.5508423, 2073.0170898, -2615.2465820, 2590.4685059
2: -628.5104370, 2068.2482910, -638.3911133, 2101.5507812, -2730.0612793, 2706.6394043
3: -888.3466797, 2269.0156250, -902.1766968, 2305.8959961, -3194.2426758, 3171.1923828
4: -1065.5307617, 2104.4018555, -1082.5723877, 2138.3459473, -3203.8767090, 3186.9736328

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0953052, upper bound: 1757.0954358
time: 1.09 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0953052, upper bound: 1757.0954358
time: 0.66 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.26 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.1148372, upper bound: 1757.1148372
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.1148372, upper bound: 1757.1152449
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.1152449, upper bound: 1757.1148372
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.1152449, upper bound: 1757.1152449
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.1149145, upper bound: 1757.1115415
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.1149145, upper bound: 1757.1115415
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.1152449, upper bound: 1757.1115415
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.1153216, upper bound: 1757.1115415
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.1115415, upper bound: 1757.1149145
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.1115415, upper bound: 1757.1153216
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.1115415, upper bound: 1757.1149145
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.1115415, upper bound: 1757.1153216
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.1115415, upper bound: 1757.1116194
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.1115415, upper bound: 1757.1116194
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.1115415, upper bound: 1757.1116194
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.1115415, upper bound: 1757.1116194
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.1135376, upper bound: 1757.0411437
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.1125668, upper bound: 1757.0308572
IS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.0366233, upper bound: 1757.0361626
IS_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.0363111, upper bound: 1757.0349349
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.1144086, upper bound: 1757.0957068
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.1144086, upper bound: 1757.0957068
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.1144086, upper bound: 1757.0957068
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.1144086, upper bound: 1757.0957068
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.1070679, upper bound: 1757.0406799
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.1067557, upper bound: 1757.0394521
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.1035587, upper bound: 1757.0405908
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.1032465, upper bound: 1757.0393631
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.0953112, upper bound: 1757.0951746
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.0953112, upper bound: 1757.0951746
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.0953112, upper bound: 1757.0951746
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.0953112, upper bound: 1757.0951746
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.0406799, upper bound: 1757.1070679
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.0405908, upper bound: 1757.1035587
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.0394521, upper bound: 1757.1067557
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.0393631, upper bound: 1757.1032465
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.0957068, upper bound: 1757.1144086
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.0957068, upper bound: 1757.1144086
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.0957068, upper bound: 1757.1144086
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.0957068, upper bound: 1757.1144086
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.0957847, upper bound: 1757.1112705
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.0957847, upper bound: 1757.1112705
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.0957847, upper bound: 1757.1112705
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.0957847, upper bound: 1757.1112705
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.0402167, upper bound: 1757.0931629
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.0402167, upper bound: 1757.0911106
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.0389890, upper bound: 1757.0928507
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.0389890, upper bound: 1757.0907983
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.0931630, upper bound: 1757.0403857
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.0928507, upper bound: 1757.0391580
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.0911265, upper bound: 1757.0403857
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.0908144, upper bound: 1757.0391580
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.0953052, upper bound: 1757.0954358
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.0953052, upper bound: 1757.0954358
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.0953052, upper bound: 1757.0954358
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1757.0953052, upper bound: 1757.0954358

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -272.6042480, 1111.4747314, -272.6042480, 1111.4747314, -1384.0788574, 1384.0788574
1: -334.2237854, 1241.4110107, -334.2237854, 1241.4110107, -1575.6347656, 1575.6347656
2: -381.0059814, 1261.2246094, -381.0059814, 1261.2246094, -1642.2304688, 1642.2304688
3: -542.2633057, 1371.0507812, -542.2633057, 1371.0507812, -1913.3140869, 1913.3140869
4: -637.6555786, 1281.7025146, -637.6555786, 1281.7025146, -1919.3580322, 1919.3580322

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1164839, upper bound: 1757.1125129
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1181179, upper bound: 1757.1149874
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -272.6042480, 1111.4747314, -311.7589417, 1265.6518555, -1538.2561035, 1423.2335205
1: -334.2237854, 1241.4110107, -381.9956665, 1413.0462646, -1747.2700195, 1623.4066162
2: -381.0059814, 1261.2246094, -436.8746033, 1436.2211914, -1817.2270508, 1698.0989990
3: -542.2633057, 1371.0507812, -620.3742065, 1564.9211426, -2107.1840820, 1991.4250488
4: -637.6555786, 1281.7025146, -733.2927246, 1460.4571533, -2098.1127930, 2014.9952393

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1164839, upper bound: 1757.1129206
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1181179, upper bound: 1757.1153950
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -311.7589417, 1265.6518555, -272.6042480, 1111.4747314, -1423.2335205, 1538.2561035
1: -381.9956665, 1413.0462646, -334.2237854, 1241.4110107, -1623.4066162, 1747.2700195
2: -436.8746033, 1436.2211914, -381.0059814, 1261.2246094, -1698.0989990, 1817.2270508
3: -620.3742065, 1564.9211426, -542.2633057, 1371.0507812, -1991.4250488, 2107.1838379
4: -733.2927246, 1460.4571533, -637.6555786, 1281.7025146, -2014.9952393, 2098.1127930

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0620115, upper bound: 1757.1119500
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1152449, upper bound: 1757.1148372
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -311.7589417, 1265.6518555, -311.7589417, 1265.6518555, -1577.4107666, 1577.4107666
1: -381.9956665, 1413.0462646, -381.9956665, 1413.0462646, -1795.0418701, 1795.0418701
2: -436.8746033, 1436.2211914, -436.8746033, 1436.2211914, -1873.0955811, 1873.0955811
3: -620.3742065, 1564.9211426, -620.3742065, 1564.9211426, -2185.2951660, 2185.2949219
4: -733.2927246, 1460.4571533, -733.2927246, 1460.4571533, -2193.7500000, 2193.7500000

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0620114, upper bound: 1757.1123576
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1152449, upper bound: 1757.1148970
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -272.6042480, 1111.4747314, -309.5700378, 1259.4353027, -1532.0395508, 1421.0445557
1: -334.2237854, 1241.4110107, -379.7421265, 1406.8647461, -1741.0885010, 1621.1530762
2: -381.0059814, 1261.2246094, -433.3220215, 1429.6942139, -1810.6999512, 1694.5466309
3: -542.2633057, 1371.0507812, -616.7457275, 1556.0572510, -2098.3203125, 1987.7963867
4: -637.6555786, 1281.7025146, -726.5832520, 1453.5422363, -2091.1977539, 2008.2857666

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1173518, upper bound: 1757.1038076
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1165610, upper bound: 1757.1092172
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1181956, upper bound: 1757.1116917
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -272.6042480, 1111.4747314, -349.7909546, 1417.5225830, -1690.1268311, 1461.2653809
1: -334.2237854, 1241.4110107, -428.7479553, 1582.5559082, -1916.7796631, 1670.1589355
2: -381.0059814, 1261.2246094, -491.1026001, 1609.0854492, -1990.0911865, 1752.3270264
3: -542.2633057, 1371.0507812, -696.9265137, 1755.5729980, -2297.8361816, 2067.9770508
4: -637.6555786, 1281.7025146, -825.4575806, 1636.9227295, -2274.5781250, 2107.1599121

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1173518, upper bound: 1757.1038076
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1165610, upper bound: 1757.1092172
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1181956, upper bound: 1757.1116917
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -311.7589417, 1265.6518555, -309.5700378, 1259.4353027, -1571.1942139, 1575.2218018
1: -381.9956665, 1413.0462646, -379.7421265, 1406.8647461, -1788.8603516, 1792.7883301
2: -436.8746033, 1436.2211914, -433.3220215, 1429.6942139, -1866.5684814, 1869.5432129
3: -620.3742065, 1564.9211426, -616.7457275, 1556.0572510, -2176.4313965, 2181.6667480
4: -733.2927246, 1460.4571533, -726.5832520, 1453.5422363, -2186.8349609, 2187.0405273

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1145855, upper bound: 1757.1036809
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0620878, upper bound: 1757.1086543
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1153216, upper bound: 1757.1115415
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -311.7589417, 1265.6518555, -349.7909546, 1417.5225830, -1729.2814941, 1615.4427490
1: -381.9956665, 1413.0462646, -428.7479553, 1582.5559082, -1964.5515137, 1841.7941895
2: -436.8746033, 1436.2211914, -491.1026001, 1609.0854492, -2045.9598389, 1927.3236084
3: -620.3742065, 1564.9211426, -696.9265137, 1755.5729980, -2375.9470215, 2261.8474121
4: -733.2927246, 1460.4571533, -825.4575806, 1636.9227295, -2370.2150879, 2285.9145508

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1145855, upper bound: 1757.1036809
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0620878, upper bound: 1757.1086543
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1153216, upper bound: 1757.1115415
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -309.5700378, 1259.4353027, -272.6042480, 1111.4747314, -1421.0445557, 1532.0395508
1: -379.7421265, 1406.8647461, -334.2237854, 1241.4110107, -1621.1530762, 1741.0885010
2: -433.3220215, 1429.6942139, -381.0059814, 1261.2246094, -1694.5466309, 1810.6999512
3: -616.7457275, 1556.0572510, -542.2633057, 1371.0507812, -1987.7963867, 2098.3203125
4: -726.5832520, 1453.5422363, -637.6555786, 1281.7025146, -2008.2857666, 2091.1977539

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1087572, upper bound: 1757.1121868
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1097402, upper bound: 1757.1121668
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -309.5700378, 1259.4353027, -311.7589417, 1265.6518555, -1575.2218018, 1571.1942139
1: -379.7421265, 1406.8647461, -381.9956665, 1413.0462646, -1792.7883301, 1788.8603516
2: -433.3220215, 1429.6942139, -436.8746033, 1436.2211914, -1869.5432129, 1866.5684814
3: -616.7457275, 1556.0572510, -620.3742065, 1564.9211426, -2181.6667480, 2176.4313965
4: -726.5832520, 1453.5422363, -733.2927246, 1460.4571533, -2187.0405273, 2186.8349609

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1087572, upper bound: 1757.1137305
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1097402, upper bound: 1757.1137105
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -349.7909546, 1417.5225830, -272.6042480, 1111.4747314, -1461.2653809, 1690.1268311
1: -428.7479553, 1582.5559082, -334.2237854, 1241.4110107, -1670.1589355, 1916.7796631
2: -491.1026001, 1609.0854492, -381.0059814, 1261.2246094, -1752.3270264, 1990.0911865
3: -696.9265137, 1755.5729980, -542.2633057, 1371.0507812, -2067.9772949, 2297.8361816
4: -825.4575806, 1636.9227295, -637.6555786, 1281.7025146, -2107.1599121, 2274.5781250

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0993630, upper bound: 1757.1118750
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1014767, upper bound: 1757.1118750
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -349.7909546, 1417.5225830, -311.7589417, 1265.6518555, -1615.4428711, 1729.2814941
1: -428.7479553, 1582.5559082, -381.9956665, 1413.0462646, -1841.7941895, 1964.5515137
2: -491.1026001, 1609.0854492, -436.8746033, 1436.2211914, -1927.3236084, 2045.9598389
3: -696.9265137, 1755.5729980, -620.3742065, 1564.9211426, -2261.8476562, 2375.9470215
4: -825.4575806, 1636.9227295, -733.2927246, 1460.4571533, -2285.9145508, 2370.2150879

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0993630, upper bound: 1757.1129030
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1014767, upper bound: 1757.1129033
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -309.5700378, 1259.4353027, -309.5700378, 1259.4353027, -1569.0051270, 1569.0051270
1: -379.7421265, 1406.8647461, -379.7421265, 1406.8647461, -1786.6069336, 1786.6069336
2: -433.3220215, 1429.6942139, -433.3220215, 1429.6942139, -1863.0161133, 1863.0161133
3: -616.7457275, 1556.0572510, -616.7457275, 1556.0572510, -2172.8029785, 2172.8029785
4: -726.5832520, 1453.5422363, -726.5832520, 1453.5422363, -2180.1252441, 2180.1252441

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1088351, upper bound: 1757.1018664
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1097402, upper bound: 1757.1018464
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -309.5700378, 1259.4353027, -349.7909546, 1417.5225830, -1727.0926514, 1609.2260742
1: -379.7421265, 1406.8647461, -428.7479553, 1582.5559082, -1962.2980957, 1835.6126709
2: -433.3220215, 1429.6942139, -491.1026001, 1609.0854492, -2042.4073486, 1920.7965088
3: -616.7457275, 1556.0572510, -696.9265137, 1755.5729980, -2372.3188477, 2252.9838867
4: -726.5832520, 1453.5422363, -825.4575806, 1636.9227295, -2363.5053711, 2278.9997559

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1088351, upper bound: 1757.1018664
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1097402, upper bound: 1757.1018464
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -349.7909546, 1417.5225830, -309.5700378, 1259.4353027, -1609.2260742, 1727.0926514
1: -428.7479553, 1582.5559082, -379.7421265, 1406.8647461, -1835.6126709, 1962.2980957
2: -491.1026001, 1609.0854492, -433.3220215, 1429.6942139, -1920.7965088, 2042.4074707
3: -696.9265137, 1755.5729980, -616.7457275, 1556.0572510, -2252.9838867, 2372.3188477
4: -825.4575806, 1636.9227295, -726.5832520, 1453.5422363, -2278.9997559, 2363.5056152

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0994409, upper bound: 1757.1015546
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1014767, upper bound: 1757.1015546
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -349.7909546, 1417.5225830, -349.7909546, 1417.5225830, -1767.3134766, 1767.3134766
1: -428.7479553, 1582.5559082, -428.7479553, 1582.5559082, -2011.3038330, 2011.3038330
2: -491.1026001, 1609.0854492, -491.1026001, 1609.0854492, -2100.1877441, 2100.1877441
3: -696.9265137, 1755.5729980, -696.9265137, 1755.5729980, -2452.4995117, 2452.4995117
4: -825.4575806, 1636.9227295, -825.4575806, 1636.9227295, -2462.3798828, 2462.3798828

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0994409, upper bound: 1757.1015546
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1014767, upper bound: 1757.1015546
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -272.6042480, 1111.4747314, -375.6163025, 1544.6511230, -1817.2553711, 1487.0908203
1: -334.2237854, 1241.4110107, -459.7278442, 1724.8203125, -2059.0434570, 1701.1386719
2: -381.0059814, 1261.2246094, -532.0983276, 1749.6722412, -2130.6782227, 1793.3228760
3: -542.2633057, 1371.0507812, -751.5830688, 1915.9863281, -2458.2492676, 2122.6337891
4: -637.6555786, 1281.7025146, -899.1909790, 1780.3447266, -2418.0002441, 2180.8935547

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1119036, upper bound: 1757.0386693
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1135376, upper bound: 1757.0411437
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -272.6042480, 1111.4747314, -392.3075867, 1609.4346924, -1882.0389404, 1503.7821045
1: -334.2237854, 1241.4110107, -480.1018982, 1796.5902100, -2130.8139648, 1721.5129395
2: -381.0059814, 1261.2246094, -554.6870728, 1824.5120850, -2205.5180664, 1815.9116211
3: -542.2633057, 1371.0507812, -783.9968872, 1995.9794922, -2538.2424316, 2155.0473633
4: -637.6555786, 1281.7025146, -936.8407593, 1855.6768799, -2493.3325195, 2218.5432129

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1109329, upper bound: 1757.0283828
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1125668, upper bound: 1757.0308572
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -272.6042480, 1111.4747314, -418.2335815, 1726.7127686, -1999.3170166, 1529.7080078
1: -334.2237854, 1241.4110107, -511.6340027, 1928.8596191, -2263.0832520, 1753.0450439
2: -381.0059814, 1261.2246094, -592.1152344, 1955.3981934, -2336.4035645, 1853.3398438
3: -542.2633057, 1371.0507812, -837.6755371, 2141.4594727, -2683.7224121, 2208.7260742
4: -637.6555786, 1281.7025146, -1002.3061523, 1988.3162842, -2625.9719238, 2284.0087891

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1173522, upper bound: 1757.0957091
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1137693, upper bound: 1757.0933431
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1178469, upper bound: 1757.0958569
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -272.6042480, 1111.4747314, -443.2770081, 1826.9315186, -2099.5354004, 1554.7517090
1: -334.2237854, 1241.4110107, -542.2296143, 2039.9180908, -2374.1413574, 1783.6406250
2: -381.0059814, 1261.2246094, -628.5104370, 2068.2482910, -2449.2536621, 1889.7347412
3: -542.2633057, 1371.0507812, -888.3466797, 2269.0156250, -2811.2788086, 2259.3972168
4: -637.6555786, 1281.7025146, -1065.5307617, 2104.4018555, -2742.0573730, 2347.2333984

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1173522, upper bound: 1757.0957091
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1137693, upper bound: 1757.0933431
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1178469, upper bound: 1757.0958569
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -311.7589417, 1265.6518555, -418.2335815, 1726.7127686, -2038.4716797, 1683.8854980
1: -381.9956665, 1413.0462646, -511.6340027, 1928.8596191, -2310.8547363, 1924.6802979
2: -436.8746033, 1436.2211914, -592.1152344, 1955.3981934, -2392.2724609, 2028.3364258
3: -620.3742065, 1564.9211426, -837.6755371, 2141.4594727, -2761.8334961, 2402.5966797
4: -733.2927246, 1460.4571533, -1002.3061523, 1988.3162842, -2721.6088867, 2462.7631836

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1139394, upper bound: 1757.0955824
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0611071, upper bound: 1757.0928109
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1144086, upper bound: 1757.0956989
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -311.7589417, 1265.6518555, -443.2770081, 1826.9315186, -2138.6901855, 1708.9288330
1: -381.9956665, 1413.0462646, -542.2296143, 2039.9180908, -2421.9128418, 1955.2758789
2: -436.8746033, 1436.2211914, -628.5104370, 2068.2482910, -2505.1225586, 2064.7316895
3: -620.3742065, 1564.9211426, -888.3466797, 2269.0156250, -2889.3898926, 2453.2678223
4: -733.2927246, 1460.4571533, -1065.5307617, 2104.4018555, -2837.6943359, 2525.9877930

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1139394, upper bound: 1757.0955824
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0611071, upper bound: 1757.0928109
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1144086, upper bound: 1757.0956989
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -309.5700378, 1259.4353027, -377.8447266, 1553.2204590, -1862.7904053, 1637.2799072
1: -379.7421265, 1406.8647461, -461.9578857, 1733.9344482, -2113.6765137, 1868.8226318
2: -433.3220215, 1429.6942139, -535.8987427, 1758.5762939, -2191.8984375, 1965.5930176
3: -616.7457275, 1556.0572510, -755.8508911, 1929.0098877, -2545.7556152, 2311.9082031
4: -726.5832520, 1453.5422363, -907.5518188, 1789.1436768, -2515.7265625, 2361.0939941

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1000078, upper bound: 1757.0270380
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1060971, upper bound: 1757.0303933
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1060083, upper bound: 1757.0404857
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1052945, upper bound: 1757.0404373
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -308.9739685, 1257.0371094, -384.8816528, 1588.1987305, -1897.1727295, 1641.9187012
1: -379.0123291, 1404.1859131, -470.1259766, 1772.4954834, -2151.5073242, 1874.3117676
2: -432.4853516, 1426.9688721, -547.4117432, 1797.7244873, -2230.2099609, 1974.3806152
3: -615.5615234, 1553.0599365, -770.8536377, 1974.8903809, -2590.4519043, 2323.9135742
4: -725.1774902, 1450.7517090, -929.3767090, 1829.8691406, -2555.0466309, 2380.1279297

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056791, upper bound: 1757.0392580
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1049653, upper bound: 1757.0392096
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -349.7909546, 1417.5225830, -378.1842957, 1554.5784912, -1904.3693848, 1795.7069092
1: -428.7479553, 1582.5559082, -462.3816833, 1735.4530029, -2164.2009277, 2044.9376221
2: -491.1026001, 1609.0854492, -536.3656006, 1760.1306152, -2251.2326660, 2145.4511719
3: -696.9265137, 1755.5729980, -756.5307007, 1930.6671143, -2627.5927734, 2512.1037598
4: -825.4575806, 1636.9227295, -908.3130493, 1790.7239990, -2616.1816406, 2545.2358398

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0936426, upper bound: 1757.0269489
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0610204, upper bound: 1757.0249941
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -349.0792542, 1414.6325684, -385.2216797, 1589.5644531, -1938.6435547, 1799.8542480
1: -427.8739624, 1579.3236084, -470.5504150, 1774.0218506, -2201.8957520, 2049.8740234
2: -490.1032715, 1605.8007812, -547.8798828, 1799.2845459, -2289.3876953, 2153.6806641
3: -695.5086670, 1751.9866943, -771.5361328, 1976.5567627, -2672.0651855, 2523.5229492
4: -823.7781982, 1633.5717773, -930.1417847, 1831.4566650, -2655.2348633, 2563.7136230

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0933304, upper bound: 1757.0255464
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0607082, upper bound: 1757.0235916
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -309.5700378, 1259.4353027, -418.2335815, 1726.7127686, -2036.2827148, 1677.6687012
1: -379.7421265, 1406.8647461, -511.6340027, 1928.8596191, -2308.6018066, 1918.4987793
2: -433.3220215, 1429.6942139, -592.1152344, 1955.3981934, -2388.7202148, 2021.8094482
3: -616.7457275, 1556.0572510, -837.6755371, 2141.4594727, -2758.2050781, 2393.7329102
4: -726.5832520, 1453.5422363, -1002.3061523, 1988.3162842, -2714.8994141, 2455.8481445

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054032, upper bound: 1757.0952443
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1041310, upper bound: 1757.0952253
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -309.5700378, 1259.4353027, -443.2770081, 1826.9315186, -2136.5014648, 1702.7122803
1: -379.7421265, 1406.8647461, -542.2296143, 2039.9180908, -2419.6599121, 1949.0943604
2: -433.3220215, 1429.6942139, -628.5104370, 2068.2482910, -2501.5703125, 2058.2045898
3: -616.7457275, 1556.0572510, -888.3466797, 2269.0156250, -2885.7612305, 2444.4038086
4: -726.5832520, 1453.5422363, -1065.5307617, 2104.4018555, -2830.9848633, 2519.0727539

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054032, upper bound: 1757.0952443
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1041310, upper bound: 1757.0952253
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -349.7909546, 1417.5225830, -418.2335815, 1726.7127686, -2076.5034180, 1835.7561035
1: -428.7479553, 1582.5559082, -511.6340027, 1928.8596191, -2357.6076660, 2094.1899414
2: -491.1026001, 1609.0854492, -592.1152344, 1955.3981934, -2446.5000000, 2201.2004395
3: -696.9265137, 1755.5729980, -837.6755371, 2141.4594727, -2838.3857422, 2593.2485352
4: -825.4575806, 1636.9227295, -1002.3061523, 1988.3162842, -2813.7739258, 2639.2285156

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0933579, upper bound: 1757.0933337
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0933334, upper bound: 1757.0933337
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -349.7909546, 1417.5225830, -443.2770081, 1826.9315186, -2176.7224121, 1860.7995605
1: -428.7479553, 1582.5559082, -542.2296143, 2039.9180908, -2468.6657715, 2124.7856445
2: -491.1026001, 1609.0854492, -628.5104370, 2068.2482910, -2559.3500977, 2237.5959473
3: -696.9265137, 1755.5729980, -888.3466797, 2269.0156250, -2965.9421387, 2643.9196777
4: -825.4575806, 1636.9227295, -1065.5307617, 2104.4018555, -2929.8591309, 2702.4531250

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0933579, upper bound: 1757.0933334
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0933334, upper bound: 1757.0933337
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -377.8447266, 1553.2204590, -309.5700378, 1259.4353027, -1637.2799072, 1862.7904053
1: -461.9578857, 1733.9344482, -379.7421265, 1406.8647461, -1868.8226318, 2113.6765137
2: -535.8987427, 1758.5762939, -433.3220215, 1429.6942139, -1965.5930176, 2191.8984375
3: -755.8508911, 1929.0098877, -616.7457275, 1556.0572510, -2311.9082031, 2545.7556152
4: -907.5518188, 1789.1436768, -726.5832520, 1453.5422363, -2361.0939941, 2515.7265625

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0405832, upper bound: 1757.1053608
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0330405, upper bound: 1757.1070621
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0271985, upper bound: 1757.1023733
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0404725, upper bound: 1757.1057966
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0406799, upper bound: 1757.1070679
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0404279, upper bound: 1757.1061882
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0400583, upper bound: 1757.1035297
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 21

Time for candidate selection: 13.35 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0406799, upper bound: 1757.1064638
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0398554, upper bound: 1757.1058508
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0376975, upper bound: 1757.1059590
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -378.1842957, 1554.5784912, -349.7909546, 1417.5225830, -1795.7069092, 1904.3693848
1: -462.3816833, 1735.4530029, -428.7479553, 1582.5559082, -2044.9376221, 2164.2006836
2: -536.3656006, 1760.1306152, -491.1026001, 1609.0854492, -2145.4511719, 2251.2326660
3: -756.5307007, 1930.6671143, -696.9265137, 1755.5729980, -2512.1037598, 2627.5927734
4: -908.3130493, 1790.7239990, -825.4575806, 1636.9227295, -2545.2358398, 2616.1816406

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0403538, upper bound: 1757.0988347
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0329514, upper bound: 1757.1035529
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0271094, upper bound: 1757.0929828
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0405908, upper bound: 1757.1035587
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0402287, upper bound: 1757.0993963
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 21

Time for candidate selection: 11.52 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0405344, upper bound: 1757.1029544
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0398359, upper bound: 1757.1029889
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0376780, upper bound: 1757.1030972
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -384.8816528, 1588.1987305, -308.9739685, 1257.0371094, -1641.9187012, 1897.1727295
1: -470.1259766, 1772.4954834, -379.0123291, 1404.1859131, -1874.3117676, 2151.5073242
2: -547.4117432, 1797.7244873, -432.4853516, 1426.9688721, -1974.3806152, 2230.2099609
3: -770.8536377, 1974.8903809, -615.5615234, 1553.0599365, -2323.9135742, 2590.4519043
4: -929.3767090, 1829.8691406, -725.1774902, 1450.7517090, -2380.1281738, 2555.0466309

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0393598, upper bound: 1757.1050439
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0314095, upper bound: 1757.1067557
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0258408, upper bound: 1757.1020554
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0392448, upper bound: 1757.1054844
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0394521, upper bound: 1757.1067558
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0392001, upper bound: 1757.1058759
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0388306, upper bound: 1757.1032175
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 21

Time for candidate selection: 12.88 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0394521, upper bound: 1757.1061224
time: 1.14 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0386277, upper bound: 1757.1056074
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0368642, upper bound: 1757.1056748
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -385.2216797, 1589.5644531, -349.0792542, 1414.6325684, -1799.8542480, 1938.6435547
1: -470.5504150, 1774.0218506, -427.8739624, 1579.3236084, -2049.8740234, 2201.8957520
2: -547.8798828, 1799.2845459, -490.1032715, 1605.8007812, -2153.6806641, 2289.3876953
3: -771.5361328, 1976.5567627, -695.5086670, 1751.9866943, -2523.5229492, 2672.0651855
4: -930.1417847, 1831.4566650, -823.7781982, 1633.5717773, -2563.7136230, 2655.2348633

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0391303, upper bound: 1757.0985179
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0313205, upper bound: 1757.1032464
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0257518, upper bound: 1757.0926649
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1756.9450496, upper bound: 1757.0944526
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0375590, upper bound: 1757.1013618
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -418.2335815, 1726.7127686, -272.6042480, 1111.4747314, -1529.7080078, 1999.3170166
1: -511.6340027, 1928.8596191, -334.2237854, 1241.4110107, -1753.0450439, 2263.0832520
2: -592.1152344, 1955.3981934, -381.0059814, 1261.2246094, -1853.3398438, 2336.4035645
3: -837.6755371, 2141.4594727, -542.2633057, 1371.0507812, -2208.7260742, 2683.7224121
4: -1002.3061523, 1988.3162842, -637.6555786, 1281.7025146, -2284.0087891, 2625.9719238

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0940284, upper bound: 1757.1093580
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0938389, upper bound: 1757.1105240
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0970547, upper bound: 1757.1112378
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -418.2335815, 1726.7127686, -311.7589417, 1265.6518555, -1683.8854980, 2038.4716797
1: -511.6340027, 1928.8596191, -381.9956665, 1413.0462646, -1924.6802979, 2310.8549805
2: -592.1152344, 1955.3981934, -436.8746033, 1436.2211914, -2028.3364258, 2392.2727051
3: -837.6755371, 2141.4594727, -620.3742065, 1564.9211426, -2402.5966797, 2761.8334961
4: -1002.3061523, 1988.3162842, -733.2927246, 1460.4571533, -2462.7631836, 2721.6088867

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0940284, upper bound: 1757.1093580
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0938389, upper bound: 1757.1105381
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0970547, upper bound: 1757.1112520
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -443.2770081, 1826.9315186, -272.6042480, 1111.4747314, -1554.7517090, 2099.5356445
1: -542.2296143, 2039.9180908, -334.2237854, 1241.4110107, -1783.6406250, 2374.1416016
2: -628.5104370, 2068.2482910, -381.0059814, 1261.2246094, -1889.7347412, 2449.2536621
3: -888.3466797, 2269.0156250, -542.2633057, 1371.0507812, -2259.3972168, 2811.2788086
4: -1065.5307617, 2104.4018555, -637.6555786, 1281.7025146, -2347.2333984, 2742.0573730

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0908843, upper bound: 1757.1105382
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0386165, upper bound: 1757.1083536
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -443.2770081, 1826.9315186, -311.7589417, 1265.6518555, -1708.9288330, 2138.6901855
1: -542.2296143, 2039.9180908, -381.9956665, 1413.0462646, -1955.2758789, 2421.9128418
2: -628.5104370, 2068.2482910, -436.8746033, 1436.2211914, -2064.7316895, 2505.1225586
3: -888.3466797, 2269.0156250, -620.3742065, 1564.9211426, -2453.2678223, 2889.3898926
4: -1065.5307617, 2104.4018555, -733.2927246, 1460.4571533, -2525.9877930, 2837.6943359

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0908843, upper bound: 1757.1110917
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0386165, upper bound: 1757.1098973
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -418.2335815, 1726.7127686, -309.5700378, 1259.4353027, -1677.6685791, 2036.2825928
1: -511.6340027, 1928.8596191, -379.7421265, 1406.8647461, -1918.4987793, 2308.6018066
2: -592.1152344, 1955.3981934, -433.3220215, 1429.6942139, -2021.8094482, 2388.7202148
3: -837.6755371, 2141.4594727, -616.7457275, 1556.0572510, -2393.7329102, 2758.2050781
4: -1002.3061523, 1988.3162842, -726.5832520, 1453.5422363, -2455.8483887, 2714.8994141

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0941063, upper bound: 1757.1012057
time: 1.04 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=2002.608154296875
rel_dist={0: [-1757.1270497407063, 1757.1270497407058]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1220434, upper bound: 1757.1064984
time: 0.76 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054749, upper bound: 1757.1054749
time: 0.69 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.57 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 0, lower bound: -1757.1220434, upper bound: 1757.1064984
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 0, lower bound: -1757.1054749, upper bound: 1757.1054749

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -356.5766602, 1445.0728760, -385.9503479, 1566.5913086, -1923.1679688, 1831.0231934
1: -437.0072937, 1613.3231201, -473.0029907, 1749.5773926, -2186.5844727, 2086.3261719
2: -500.5809937, 1640.3117676, -541.3590088, 1778.4146729, -2278.9956055, 2181.6706543
3: -710.3735962, 1789.6778564, -768.5867920, 1937.9490967, -2648.3225098, 2558.2644043
4: -841.4490967, 1668.5886230, -909.2456665, 1808.5659180, -2650.0151367, 2577.8342285

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054186, upper bound: 1757.1054186
time: 0.72 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054186, upper bound: 1757.1054736
time: 0.68 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -457.7469177, 1887.2629395, -381.3312378, 1551.9245605, -2009.6715088, 2268.5939941
1: -559.7200317, 2107.2399902, -467.3357239, 1733.1770020, -2292.8964844, 2574.5756836
2: -649.0897827, 2136.3076172, -535.2318115, 1761.7320557, -2410.8217773, 2671.5395508
3: -917.2413330, 2344.1276855, -759.7696533, 1919.2121582, -2836.4528809, 3103.8967285
4: -1100.8392334, 2173.5678711, -899.2909546, 1791.4243164, -2892.2636719, 3072.8586426

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054736, upper bound: 1757.1054194
time: 0.71 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054736, upper bound: 1757.1054749
time: 0.73 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.75 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.75
Output dim: 0, lower bound: -1757.1054186, upper bound: 1757.1054186
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.75
Output dim: 0, lower bound: -1757.1054186, upper bound: 1757.1054736
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.75
Output dim: 0, lower bound: -1757.1054736, upper bound: 1757.1054194
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.75
Output dim: 0, lower bound: -1757.1054736, upper bound: 1757.1054749

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -356.5766602, 1445.0728760, -356.5766602, 1445.0728760, -1801.6495361, 1801.6495361
1: -437.0072937, 1613.3231201, -437.0072937, 1613.3231201, -2050.3303223, 2050.3303223
2: -500.5809937, 1640.3117676, -500.5809937, 1640.3117676, -2140.8925781, 2140.8928223
3: -710.3735962, 1789.6778564, -710.3735962, 1789.6778564, -2500.0515137, 2500.0515137
4: -841.4490967, 1668.5886230, -841.4490967, 1668.5886230, -2510.0375977, 2510.0375977

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1220434, upper bound: 1757.0929224
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1219706, upper bound: 1757.1064631
time: 0.81 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -356.5766602, 1445.0728760, -457.7469177, 1887.2629395, -2243.8393555, 1902.8198242
1: -437.0072937, 1613.3231201, -559.7200317, 2107.2399902, -2544.2473145, 2173.0424805
2: -500.5809937, 1640.3117676, -649.0897827, 2136.3076172, -2636.8884277, 2289.4016113
3: -710.3735962, 1789.6778564, -917.2413330, 2344.1276855, -3054.5012207, 2706.9191895
4: -841.4490967, 1668.5886230, -1100.8392334, 2173.5678711, -3015.0170898, 2769.4277344

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1220434, upper bound: 1757.0929224
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1219706, upper bound: 1757.1064984
time: 0.71 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -457.7469177, 1887.2629395, -356.5766602, 1445.0728760, -1902.8198242, 2243.8393555
1: -559.7200317, 2107.2399902, -437.0072937, 1613.3231201, -2173.0427246, 2544.2473145
2: -649.0897827, 2136.3076172, -500.5809937, 1640.3117676, -2289.4016113, 2636.8886719
3: -917.2413330, 2344.1276855, -710.3735962, 1789.6778564, -2706.9191895, 3054.5012207
4: -1100.8392334, 2173.5678711, -841.4490967, 1668.5886230, -2769.4277344, 3015.0170898

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0398435, upper bound: 1757.0889633
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054736, upper bound: 1757.1054194
time: 0.78 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -457.7469177, 1887.2629395, -457.7469177, 1887.2629395, -2345.0095215, 2345.0095215
1: -559.7200317, 2107.2399902, -559.7200317, 2107.2399902, -2666.9597168, 2666.9597168
2: -649.0897827, 2136.3076172, -649.0897827, 2136.3076172, -2785.3974609, 2785.3974609
3: -917.2413330, 2344.1276855, -917.2413330, 2344.1276855, -3261.3691406, 3261.3691406
4: -1100.8392334, 2173.5678711, -1100.8392334, 2173.5678711, -3274.4072266, 3274.4072266

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0398435, upper bound: 1757.0889633
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054736, upper bound: 1757.1054690
time: 0.73 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.77 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.77
Output dim: 0, lower bound: -1757.1220434, upper bound: 1757.0929224
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.77
Output dim: 0, lower bound: -1757.1219706, upper bound: 1757.1064631
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.77
Output dim: 0, lower bound: -1757.1220434, upper bound: 1757.0929224
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.77
Output dim: 0, lower bound: -1757.1219706, upper bound: 1757.1064984
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.77
Output dim: 0, lower bound: -1757.0398435, upper bound: 1757.0889633
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.77
Output dim: 0, lower bound: -1757.1054736, upper bound: 1757.1054194
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.77
Output dim: 0, lower bound: -1757.0398435, upper bound: 1757.0889633
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.77
Output dim: 0, lower bound: -1757.1054736, upper bound: 1757.1054690

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -314.7524109, 1277.9980469, -347.7778625, 1408.9268799, -1723.6791992, 1625.7758789
1: -385.6337891, 1426.8695068, -426.3086853, 1572.9467773, -1958.5802002, 1853.1782227
2: -441.0567932, 1450.1931152, -488.1355591, 1599.3753662, -2040.4321289, 1938.3286133
3: -626.3041992, 1580.1624756, -692.7955322, 1745.0400391, -2371.3442383, 2272.9580078
4: -740.3444824, 1474.6290283, -820.3388062, 1627.0577393, -2367.4016113, 2294.9677734

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1233874, upper bound: 1757.1233874
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1233874, upper bound: 1757.1233874
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -352.7528076, 1429.8717041, -355.2608032, 1439.8211670, -1792.5739746, 1785.1325684
1: -432.3590393, 1596.3819580, -435.4067688, 1607.4708252, -2039.8293457, 2031.7886963
2: -495.2670288, 1623.0344238, -498.7470398, 1634.3421631, -2129.6091309, 2121.7814941
3: -702.8384399, 1770.8116455, -707.7752075, 1783.1641846, -2486.0021973, 2478.5866699
4: -832.5136719, 1651.0700684, -838.3652954, 1662.5323486, -2495.0458984, 2489.4353027

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1233874, upper bound: 1757.1236659
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1233874, upper bound: 1757.1236659
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -314.7524109, 1277.9980469, -450.5782166, 1858.2020264, -2172.9538574, 1728.5762939
1: -385.6337891, 1426.8695068, -550.8789673, 2074.7753906, -2460.4091797, 1977.7482910
2: -441.0567932, 1450.1931152, -639.0380249, 2103.3046875, -2544.3615723, 2089.2309570
3: -626.3041992, 1580.1624756, -902.9243164, 2308.4252930, -2934.7292480, 2483.0864258
4: -740.3444824, 1474.6290283, -1084.1385498, 2140.0996094, -2880.4440918, 2558.7670898

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1026017, upper bound: 1757.0402457
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1026017, upper bound: 1757.0929224
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -352.7528076, 1429.8717041, -455.2709351, 1877.2188721, -2229.9716797, 1885.1425781
1: -432.3590393, 1596.3819580, -556.7249146, 2096.0356445, -2528.3942871, 2153.1069336
2: -495.2670288, 1623.0344238, -645.5795898, 2124.9370117, -2620.2041016, 2268.6140137
3: -702.8384399, 1770.8116455, -912.3128052, 2331.5974121, -3034.4355469, 2683.1242676
4: -832.5136719, 1651.0700684, -1094.8269043, 2162.0451660, -2994.5585938, 2745.8969727

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1092402, upper bound: 1757.0973798
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1102104, upper bound: 1757.0954807
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -393.3483887, 1617.7186279, -347.7778625, 1408.9268799, -1802.2752686, 1965.4964600
1: -481.2612000, 1806.1429443, -426.3086853, 1572.9467773, -2054.2080078, 2232.4514160
2: -557.5192261, 1832.2569580, -488.1355591, 1599.3753662, -2156.8942871, 2320.3925781
3: -787.0695801, 2007.7425537, -692.7955322, 1745.0400391, -2532.1096191, 2700.5378418
4: -943.0974121, 1864.2763672, -820.3388062, 1627.0577393, -2570.1540527, 2684.6152344

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0402457, upper bound: 1757.1026017
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0402457, upper bound: 1757.1026017
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -450.1930237, 1856.5698242, -355.2608032, 1439.8211670, -1890.0140381, 2211.8303223
1: -550.5508423, 2073.0170898, -435.4067688, 1607.4708252, -2158.0214844, 2508.4235840
2: -638.3911133, 2101.5507812, -498.7470398, 1634.3421631, -2272.7333984, 2600.2978516
3: -902.1766968, 2305.8959961, -707.7752075, 1783.1641846, -2685.3405762, 3013.6711426
4: -1082.5723877, 2138.3459473, -838.3652954, 1662.5323486, -2745.1040039, 2976.7111816

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0929224, upper bound: 1757.1219706
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0929224, upper bound: 1757.1219706
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -393.3483887, 1617.7186279, -450.5782166, 1858.2020264, -2251.5495605, 2068.2968750
1: -481.2612000, 1806.1429443, -550.8789673, 2074.7753906, -2556.0366211, 2357.0217285
2: -557.5192261, 1832.2569580, -639.0380249, 2103.3046875, -2660.8237305, 2471.2949219
3: -787.0695801, 2007.7425537, -902.9243164, 2308.4252930, -3095.4948730, 2910.6662598
4: -943.0974121, 1864.2763672, -1084.1385498, 2140.0996094, -3083.1965332, 2948.4150391

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0360244, upper bound: 1757.0360244
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0360244, upper bound: 1757.0889633
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -450.1930237, 1856.5698242, -455.2709351, 1877.2188721, -2327.4118652, 2311.8400879
1: -550.5508423, 2073.0170898, -556.7249146, 2096.0356445, -2646.5864258, 2629.7419434
2: -638.3911133, 2101.5507812, -645.5795898, 2124.9370117, -2763.3281250, 2747.1303711
3: -902.1766968, 2305.8959961, -912.3128052, 2331.5974121, -3233.7739258, 3218.2087402
4: -1082.5723877, 2138.3459473, -1094.8269043, 2162.0451660, -3244.6169434, 3233.1728516

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0951766, upper bound: 1757.0972501
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0951675, upper bound: 1757.0952896
time: 0.81 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.80 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 0, lower bound: -1757.1233874, upper bound: 1757.1233874
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 0, lower bound: -1757.1233874, upper bound: 1757.1233874
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 0, lower bound: -1757.1233874, upper bound: 1757.1236659
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 0, lower bound: -1757.1233874, upper bound: 1757.1236659
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 0, lower bound: -1757.1026017, upper bound: 1757.0402457
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 0, lower bound: -1757.1026017, upper bound: 1757.0929224
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 0, lower bound: -1757.1092402, upper bound: 1757.0973798
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 0, lower bound: -1757.1102104, upper bound: 1757.0954807
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 0, lower bound: -1757.0402457, upper bound: 1757.1026017
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 0, lower bound: -1757.0402457, upper bound: 1757.1026017
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 0, lower bound: -1757.0929224, upper bound: 1757.1219706
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 0, lower bound: -1757.0929224, upper bound: 1757.1219706
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.80
Output dim: 0, lower bound: -1757.0360244, upper bound: 1757.0360244
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 0, lower bound: -1757.0360244, upper bound: 1757.0889633
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 0, lower bound: -1757.0951766, upper bound: 1757.0972501
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 0, lower bound: -1757.0951675, upper bound: 1757.0952896

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -314.7524109, 1277.9980469, -314.7524109, 1277.9980469, -1592.7503662, 1592.7503662
1: -385.6337891, 1426.8695068, -385.6337891, 1426.8695068, -1812.5030518, 1812.5030518
2: -441.0567932, 1450.1931152, -441.0567932, 1450.1931152, -1891.2498779, 1891.2498779
3: -626.3041992, 1580.1624756, -626.3041992, 1580.1624756, -2206.4663086, 2206.4663086
4: -740.3444824, 1474.6290283, -740.3444824, 1474.6290283, -2214.9733887, 2214.9731445

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1177641, upper bound: 1757.1114408
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1148268, upper bound: 1757.1112526
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -314.7524109, 1277.9980469, -352.6034851, 1429.2711182, -1744.0234375, 1630.6014404
1: -385.6337891, 1426.8695068, -432.1757507, 1595.7115479, -1981.3450928, 1859.0452881
2: -441.0567932, 1450.1931152, -495.0612183, 1622.3526611, -2063.4094238, 1945.2543945
3: -626.3041992, 1580.1624756, -702.5424805, 1770.0714111, -2396.3754883, 2282.7045898
4: -740.3444824, 1474.6290283, -832.1697388, 1650.3776855, -2390.7221680, 2306.7988281

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1177641, upper bound: 1757.1114408
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1148268, upper bound: 1757.1112526
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -352.7528076, 1429.8717041, -314.7524109, 1277.9980469, -1630.7508545, 1744.6241455
1: -432.3590393, 1596.3819580, -385.6337891, 1426.8695068, -1859.2283936, 1982.0156250
2: -495.2670288, 1623.0344238, -441.0567932, 1450.1931152, -1945.4600830, 2064.0913086
3: -702.8384399, 1770.8116455, -626.3041992, 1580.1624756, -2283.0007324, 2397.1157227
4: -832.5136719, 1651.0700684, -740.3444824, 1474.6290283, -2307.1425781, 2391.4145508

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1179918, upper bound: 1757.1115983
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1112526, upper bound: 1757.1113094
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -352.7528076, 1429.8717041, -352.7528076, 1429.8717041, -1782.6245117, 1782.6245117
1: -432.3590393, 1596.3819580, -432.3590393, 1596.3819580, -2028.7409668, 2028.7409668
2: -495.2670288, 1623.0344238, -495.2670288, 1623.0344238, -2118.3015137, 2118.3015137
3: -702.8384399, 1770.8116455, -702.8384399, 1770.8116455, -2473.6499023, 2473.6499023
4: -832.5136719, 1651.0700684, -832.5136719, 1651.0700684, -2483.5837402, 2483.5837402

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1179918, upper bound: 1757.1115983
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1112526, upper bound: 1757.1113094
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -314.7524109, 1277.9980469, -393.3483887, 1617.7186279, -1932.4709473, 1671.3464355
1: -385.6337891, 1426.8695068, -481.2612000, 1806.1429443, -2191.7768555, 1908.1303711
2: -441.0567932, 1450.1931152, -557.5192261, 1832.2569580, -2273.3137207, 2007.7124023
3: -626.3041992, 1580.1624756, -787.0695801, 2007.7425537, -2634.0463867, 2367.2319336
4: -740.3444824, 1474.6290283, -943.0974121, 1864.2763672, -2604.6208496, 2417.7255859

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1077633, upper bound: 1757.0398151
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1059922, upper bound: 1757.0400414
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -314.7524109, 1277.9980469, -450.1930237, 1856.5698242, -2171.3222656, 1728.1907959
1: -385.6337891, 1426.8695068, -550.5508423, 2073.0170898, -2458.6503906, 1977.4204102
2: -441.0567932, 1450.1931152, -638.3911133, 2101.5507812, -2542.6076660, 2088.5842285
3: -626.3041992, 1580.1624756, -902.1766968, 2305.8959961, -2932.2001953, 2482.3388672
4: -740.3444824, 1474.6290283, -1082.5723877, 2138.3459473, -2878.6904297, 2557.2004395

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1077633, upper bound: 1757.0926452
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1059922, upper bound: 1757.0926143
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -347.3101807, 1407.9709473, -422.9389038, 1746.0081787, -2093.3178711, 1830.9096680
1: -425.6969910, 1571.9342041, -517.3671265, 1950.3825684, -2376.0795898, 2089.3012695
2: -487.6969910, 1598.1976318, -598.7854614, 1977.2226562, -2464.9196777, 2196.9831543
3: -692.0366211, 1743.7960205, -847.0921631, 2165.4328613, -2857.4694824, 2590.8879395
4: -819.8892212, 1625.7675781, -1013.6734619, 2010.4298096, -2830.3190918, 2639.4406738

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1092402, upper bound: 1757.0954807
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1092402, upper bound: 1757.0954807
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -351.8849487, 1426.2396240, -448.3282471, 1847.5351562, -2199.4201660, 1874.5678711
1: -431.3024902, 1592.3155518, -548.3832397, 2062.8842773, -2494.1865234, 2140.6987305
2: -494.0453491, 1618.9344482, -635.6739502, 2091.5781250, -2585.6235352, 2254.6081543
3: -701.1049194, 1766.3093262, -898.4569702, 2294.6586914, -2995.7636719, 2664.7658691
4: -830.4378662, 1646.9124756, -1077.7500000, 2128.0458984, -2958.4836426, 2724.6623535

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1102104, upper bound: 1757.0954807
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1102104, upper bound: 1757.0954807
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -393.3483887, 1617.7186279, -314.7524109, 1277.9980469, -1671.3464355, 1932.4709473
1: -481.2612000, 1806.1429443, -385.6337891, 1426.8695068, -1908.1303711, 2191.7768555
2: -557.5192261, 1832.2569580, -441.0567932, 1450.1931152, -2007.7124023, 2273.3137207
3: -787.0695801, 2007.7425537, -626.3041992, 1580.1624756, -2367.2319336, 2634.0463867
4: -943.0974121, 1864.2763672, -740.3444824, 1474.6290283, -2417.7255859, 2604.6208496

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0326063, upper bound: 1757.1020152
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0357755, upper bound: 1757.0362219
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0345447, upper bound: 1757.0359080
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -393.3483887, 1617.7186279, -352.6034851, 1429.2711182, -1822.6195068, 1970.3218994
1: -481.2612000, 1806.1429443, -432.1757507, 1595.7115479, -2076.9726562, 2238.3186035
2: -557.5192261, 1832.2569580, -495.0612183, 1622.3526611, -2179.8718262, 2327.3181152
3: -787.0695801, 2007.7425537, -702.5424805, 1770.0714111, -2557.1411133, 2710.2844238
4: -943.0974121, 1864.2763672, -832.1697388, 1650.3776855, -2593.4748535, 2696.4460449

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0326063, upper bound: 1757.1020152
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0357755, upper bound: 1757.0998260
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0345447, upper bound: 1757.0996288
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -450.1930237, 1856.5698242, -314.7524109, 1277.9980469, -1728.1907959, 2171.3220215
1: -550.5508423, 2073.0170898, -385.6337891, 1426.8695068, -1977.4204102, 2458.6503906
2: -638.3911133, 2101.5507812, -441.0567932, 1450.1931152, -2088.5842285, 2542.6076660
3: -902.1766968, 2305.8959961, -626.3041992, 1580.1624756, -2482.3388672, 2932.2001953
4: -1082.5723877, 2138.3459473, -740.3444824, 1474.6290283, -2557.2004395, 2878.6904297

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0855558, upper bound: 1757.1161638
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0384798, upper bound: 1757.0993833
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -450.1930237, 1856.5698242, -352.7528076, 1429.8717041, -1880.0645752, 2209.3227539
1: -550.5508423, 2073.0170898, -432.3590393, 1596.3819580, -2146.9328613, 2505.3757324
2: -638.3911133, 2101.5507812, -495.2670288, 1623.0344238, -2261.4255371, 2596.8178711
3: -902.1766968, 2305.8959961, -702.8384399, 1770.8116455, -2672.9882812, 3008.7343750
4: -1082.5723877, 2138.3459473, -832.5136719, 1651.0700684, -2733.6418457, 2970.8596191

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0855558, upper bound: 1757.1161638
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0384798, upper bound: 1757.0993831
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -393.3483887, 1617.7186279, -450.1930237, 1856.5698242, -2249.9177246, 2067.9116211
1: -481.2612000, 1806.1429443, -550.5508423, 2073.0170898, -2554.2778320, 2356.6938477
2: -557.5192261, 1832.2569580, -638.3911133, 2101.5507812, -2659.0700684, 2470.6479492
3: -787.0695801, 2007.7425537, -902.1766968, 2305.8959961, -3092.9655762, 2909.9189453
4: -943.0974121, 1864.2763672, -1082.5723877, 2138.3459473, -3081.4428711, 2946.8481445

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0357109, upper bound: 1757.0889633
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0344802, upper bound: 1757.0886589
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -447.0111389, 1843.9216309, -422.9389038, 1746.0081787, -2193.0192871, 2266.8605957
1: -546.6158447, 2058.8889160, -517.3671265, 1950.3825684, -2496.9982910, 2576.2556152
2: -633.9902344, 2087.1635742, -598.7854614, 1977.2226562, -2611.2128906, 2685.9489746
3: -895.8502808, 2290.3886719, -847.0921631, 2165.4328613, -3061.2827148, 3137.4804688
4: -1075.3002930, 2123.7011719, -1013.6734619, 2010.4298096, -3085.7299805, 3137.3745117

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0951675, upper bound: 1757.0952896
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0951675, upper bound: 1757.0952896
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -448.0812378, 1847.5814209, -448.3282471, 1847.5351562, -2295.6164551, 2295.9094238
1: -548.0209351, 2062.9819336, -548.3832397, 2062.8842773, -2610.9052734, 2611.3652344
2: -635.3887939, 2091.4255371, -635.6739502, 2091.5781250, -2726.9667969, 2727.0988770
3: -897.9732056, 2294.7104492, -898.4569702, 2294.6586914, -3192.6318359, 3193.1672363
4: -1077.3940430, 2128.0505371, -1077.7500000, 2128.0458984, -3205.4399414, 3205.8005371

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0858560, upper bound: 1757.0378543
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0349694, upper bound: 1757.0349694
time: 0.68 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.81 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1757.1177641, upper bound: 1757.1114408
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1757.1148268, upper bound: 1757.1112526
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1757.1177641, upper bound: 1757.1114408
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1757.1148268, upper bound: 1757.1112526
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1757.1179918, upper bound: 1757.1115983
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1757.1112526, upper bound: 1757.1113094
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1757.1179918, upper bound: 1757.1115983
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1757.1112526, upper bound: 1757.1113094
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1757.1077633, upper bound: 1757.0398151
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1757.1059922, upper bound: 1757.0400414
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1757.1077633, upper bound: 1757.0926452
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1757.1059922, upper bound: 1757.0926143
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1757.1092402, upper bound: 1757.0954807
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1757.1092402, upper bound: 1757.0954807
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1757.1102104, upper bound: 1757.0954807
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1757.1102104, upper bound: 1757.0954807
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.81
Output dim: 0, lower bound: -1757.0357755, upper bound: 1757.0362219
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.81
Output dim: 0, lower bound: -1757.0345447, upper bound: 1757.0359080
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1757.0357755, upper bound: 1757.0998260
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1757.0345447, upper bound: 1757.0996288
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1757.0855558, upper bound: 1757.1161638
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1757.0384798, upper bound: 1757.0993833
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1757.0855558, upper bound: 1757.1161638
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1757.0384798, upper bound: 1757.0993831
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1757.0357109, upper bound: 1757.0889633
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1757.0344802, upper bound: 1757.0886589
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1757.0951675, upper bound: 1757.0952896
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1757.0951675, upper bound: 1757.0952896
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1757.0858560, upper bound: 1757.0378543
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.81
Output dim: 0, lower bound: -1757.0349694, upper bound: 1757.0349694

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -272.6042480, 1111.4747314, -309.7560730, 1257.9038086, -1530.5080566, 1421.2307129
1: -334.2237854, 1241.4110107, -379.5365295, 1404.4398193, -1738.6635742, 1620.9475098
2: -381.0059814, 1261.2246094, -434.1558838, 1427.3826904, -1808.3885498, 1695.3804932
3: -542.2633057, 1371.0507812, -616.4439087, 1555.3936768, -2097.6564941, 1987.4945068
4: -637.6555786, 1281.7025146, -728.9026489, 1451.3978271, -2089.0534668, 2010.6051025

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1145687, upper bound: 1757.1145687
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1145687, upper bound: 1757.1148268
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -311.7589417, 1265.6518555, -313.8494873, 1274.2600098, -1586.0189209, 1579.5012207
1: -381.9956665, 1413.0462646, -384.5373535, 1422.6838379, -1804.6794434, 1797.5834961
2: -436.8746033, 1436.2211914, -439.7952881, 1445.9663086, -1882.8404541, 1876.0163574
3: -620.3742065, 1564.9211426, -624.5125122, 1575.5447998, -2195.9189453, 2189.4333496
4: -733.2927246, 1460.4571533, -738.2120972, 1470.3409424, -2203.6335449, 2198.6691895

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1148268, upper bound: 1757.1145687
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1148268, upper bound: 1757.1148268
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -272.6042480, 1111.4747314, -347.1562805, 1407.3524170, -1679.9566650, 1458.6309814
1: -334.2237854, 1241.4110107, -425.5080261, 1571.2438965, -1905.4676514, 1666.9189453
2: -381.0059814, 1261.2246094, -487.4848022, 1597.4956055, -1978.5015869, 1748.7094727
3: -542.2633057, 1371.0507812, -691.7315674, 1743.0355225, -2285.2988281, 2062.7822266
4: -637.6555786, 1281.7025146, -819.5352783, 1625.0538330, -2262.7094727, 2101.2377930

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1146023, upper bound: 1757.1112526
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1146023, upper bound: 1757.1112526
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -311.7589417, 1265.6518555, -351.7384949, 1425.6505127, -1737.4094238, 1617.3903809
1: -381.9956665, 1413.0462646, -431.1226807, 1591.6577148, -1973.6533203, 1844.1689453
2: -436.8746033, 1436.2211914, -493.8435364, 1618.2659912, -2055.1401367, 1930.0646973
3: -620.3742065, 1564.9211426, -700.8145752, 1765.5847168, -2385.9587402, 2265.7351074
4: -733.2927246, 1460.4571533, -830.1007690, 1646.2331543, -2379.5251465, 2290.5578613

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1148268, upper bound: 1757.1112526
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1148268, upper bound: 1757.1112526
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -309.5700378, 1259.4353027, -309.7560730, 1257.9038086, -1567.4737549, 1569.1912842
1: -379.7421265, 1406.8647461, -379.5365295, 1404.4398193, -1784.1818848, 1786.4012451
2: -433.3220215, 1429.6942139, -434.1558838, 1427.3826904, -1860.7047119, 1863.8499756
3: -616.7457275, 1556.0572510, -616.4439087, 1555.3936768, -2172.1394043, 2172.5012207
4: -726.5832520, 1453.5422363, -728.9026489, 1451.3978271, -2177.9809570, 2182.4448242

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1135677, upper bound: 1757.1128473
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1112526, upper bound: 1757.1146023
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1112526, upper bound: 1757.1148268
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -349.7909546, 1417.5225830, -313.8494873, 1274.2600098, -1624.0510254, 1731.3720703
1: -428.7479553, 1582.5559082, -384.5373535, 1422.6838379, -1851.4317627, 1967.0931396
2: -491.1026001, 1609.0854492, -439.7952881, 1445.9663086, -1937.0684814, 2048.8808594
3: -696.9265137, 1755.5729980, -624.5125122, 1575.5447998, -2272.4711914, 2380.0849609
4: -825.4575806, 1636.9227295, -738.2120972, 1470.3409424, -2295.7983398, 2375.1347656

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1033073, upper bound: 1757.1121398
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1112526, upper bound: 1757.1146023
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1112526, upper bound: 1757.1148268
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -309.5700378, 1259.4353027, -347.3101807, 1407.9709473, -1717.5407715, 1606.7453613
1: -379.7421265, 1406.8647461, -425.6969910, 1571.9342041, -1951.6762695, 1832.5617676
2: -433.3220215, 1429.6942139, -487.6969910, 1598.1976318, -2031.5196533, 1917.3909912
3: -616.7457275, 1556.0572510, -692.0366211, 1743.7960205, -2360.5417480, 2248.0937500
4: -726.5832520, 1453.5422363, -819.8892212, 1625.7675781, -2352.3503418, 2273.4313965

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1112526, upper bound: 1757.1113094
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1112526, upper bound: 1757.1113094
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -349.7909546, 1417.5225830, -351.8849487, 1426.2396240, -1776.0305176, 1769.4074707
1: -428.7479553, 1582.5559082, -431.3024902, 1592.3155518, -2021.0634766, 2013.8583984
2: -491.1026001, 1609.0854492, -494.0453491, 1618.9344482, -2110.0366211, 2103.1308594
3: -696.9265137, 1755.5729980, -701.1049194, 1766.3093262, -2463.2355957, 2456.6779785
4: -825.4575806, 1636.9227295, -830.4378662, 1646.9124756, -2472.3698730, 2467.3601074

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1112526, upper bound: 1757.1113094
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1112526, upper bound: 1757.1113094
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -312.4457397, 1267.8625488, -392.5877380, 1614.5740967, -1927.0197754, 1660.4503174
1: -382.8611450, 1415.5452881, -480.3318176, 1802.6135254, -2185.4743652, 1895.8769531
2: -437.6739197, 1438.7808838, -556.4561768, 1828.6881104, -2266.3620605, 1995.2369385
3: -621.6668701, 1567.5035400, -785.5613403, 2003.8983154, -2625.5651855, 2353.0649414
4: -734.3968506, 1463.1627197, -941.3337402, 1860.6394043, -2595.0361328, 2404.4958496

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1076013, upper bound: 1757.0321757
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0974902, upper bound: 1757.0261368
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1077633, upper bound: 1757.0398151
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056648, upper bound: 1757.0398151
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1077633, upper bound: 1757.0398151
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1029951, upper bound: 1757.0286552
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -312.1175537, 1267.4616699, -392.9958191, 1616.2789307, -1928.3964844, 1660.4575195
1: -382.4297180, 1415.0886230, -480.8335571, 1804.5328369, -2186.9624023, 1895.9221191
2: -437.3249207, 1438.2316895, -557.0274658, 1830.6243896, -2267.9489746, 1995.2591553
3: -621.0979614, 1567.0180664, -786.3687744, 2005.9515381, -2627.0488281, 2353.3867188
4: -733.9794312, 1462.5063477, -942.2601929, 1862.6179199, -2596.5974121, 2404.7661133

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1057757, upper bound: 1757.0324020
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0360519, upper bound: 1757.0355836
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0357210, upper bound: 1757.0343530
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -312.4457397, 1267.8625488, -449.3885803, 1853.2795410, -2165.7246094, 1717.2508545
1: -382.8611450, 1415.5452881, -549.5630493, 2069.3320312, -2452.1931152, 1965.1082764
2: -437.6739197, 1438.7808838, -637.2600708, 2097.8129883, -2535.4868164, 2076.0410156
3: -621.6668701, 1567.5035400, -900.5806885, 2301.8732910, -2923.5400391, 2468.0842285
4: -734.3968506, 1463.1627197, -1080.6976318, 2134.5661621, -2868.9628906, 2543.8603516

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1164592, upper bound: 1757.0852769
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1075388, upper bound: 1757.0382816
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -312.1175537, 1267.4616699, -449.7668762, 1854.8229980, -2166.9401855, 1717.2285156
1: -382.4297180, 1415.0886230, -550.0299683, 2071.0649414, -2453.4946289, 1965.1184082
2: -437.3249207, 1438.2316895, -637.7961426, 2099.5671387, -2536.8920898, 2076.0270996
3: -621.0979614, 1567.0180664, -901.3229370, 2303.7238770, -2924.8215332, 2468.3410645
4: -733.9794312, 1462.5063477, -1081.5584717, 2136.3266602, -2870.3061523, 2544.0646973

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1052588, upper bound: 1757.0852387
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1020657, upper bound: 1757.0382655
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -309.5700378, 1259.4353027, -422.9389038, 1746.0081787, -2055.5781250, 1682.3740234
1: -379.7421265, 1406.8647461, -517.3671265, 1950.3825684, -2330.1247559, 1924.2319336
2: -433.3220215, 1429.6942139, -598.7854614, 1977.2226562, -2410.5446777, 2028.4793701
3: -616.7457275, 1556.0572510, -847.0921631, 2165.4328613, -2782.1787109, 2403.1491699
4: -726.5832520, 1453.5422363, -1013.6734619, 2010.4298096, -2737.0131836, 2467.2153320

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1044493, upper bound: 1757.0935739
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1085965, upper bound: 1757.0966595
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -349.7909546, 1417.5225830, -422.9389038, 1746.0081787, -2095.7990723, 1840.4614258
1: -428.7479553, 1582.5559082, -517.3671265, 1950.3825684, -2379.1306152, 2099.9230957
2: -491.1026001, 1609.0854492, -598.7854614, 1977.2226562, -2468.3251953, 2207.8708496
3: -696.9265137, 1755.5729980, -847.0921631, 2165.4328613, -2862.3593750, 2602.6647949
4: -825.4575806, 1636.9227295, -1013.6734619, 2010.4298096, -2835.8874512, 2650.5959473

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1044493, upper bound: 1757.0935741
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1085965, upper bound: 1757.0966595
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -309.5700378, 1259.4353027, -448.3282471, 1847.5351562, -2157.1052246, 1707.7634277
1: -379.7421265, 1406.8647461, -548.3832397, 2062.8842773, -2442.6262207, 1955.2480469
2: -433.3220215, 1429.6942139, -635.6739502, 2091.5781250, -2524.9001465, 2065.3676758
3: -616.7457275, 1556.0572510, -898.4569702, 2294.6586914, -2911.4042969, 2454.5139160
4: -726.5832520, 1453.5422363, -1077.7500000, 2128.0458984, -2854.6289062, 2531.2919922

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1092402, upper bound: 1757.0954749
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1007618, upper bound: 1757.0896254
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1092399, upper bound: 1757.0954807
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -349.7909546, 1417.5225830, -448.3282471, 1847.5351562, -2197.3261719, 1865.8508301
1: -428.7479553, 1582.5559082, -548.3832397, 2062.8842773, -2491.6320801, 2130.9392090
2: -491.1026001, 1609.0854492, -635.6739502, 2091.5781250, -2582.6801758, 2244.7590332
3: -696.9265137, 1755.5729980, -898.4569702, 2294.6586914, -2991.5849609, 2654.0297852
4: -825.4575806, 1636.9227295, -1077.7500000, 2128.0458984, -2953.5034180, 2714.6723633

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1092399, upper bound: 1757.0954749
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1007618, upper bound: 1757.0896254
time: 1.17 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1092402, upper bound: 1757.0954807
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -378.1842957, 1554.5784912, -348.2732544, 1411.3165283, -1789.5008545, 1902.8518066
1: -462.3816833, 1735.4530029, -426.8919067, 1575.6802979, -2038.0620117, 2162.3447266
2: -536.3656006, 1760.1306152, -489.0858154, 1601.9511719, -2138.3168945, 2249.2160645
3: -756.5307007, 1930.6671143, -693.9682617, 1748.1473389, -2504.6774902, 2624.6345215
4: -908.3130493, 1790.7239990, -822.1591187, 1629.6982422, -2538.0112305, 2612.8830566

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0265817, upper bound: 1757.0895696
time: 0.96 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0248932, upper bound: 1757.0607556
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -385.2216797, 1589.5644531, -347.3331909, 1407.9455566, -1793.1672363, 1936.8975830
1: -470.5504150, 1774.0218506, -425.7163391, 1571.8781738, -2042.4284668, 2199.7382812
2: -547.8798828, 1799.2845459, -487.6546936, 1598.1159668, -2145.9958496, 2286.9389648
3: -771.5361328, 1976.5567627, -692.0510864, 1743.5112305, -2515.0473633, 2668.6076660
4: -930.1417847, 1831.4566650, -819.6889648, 1625.6486816, -2555.7905273, 2651.1455078

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0251848, upper bound: 1757.0893851
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0234862, upper bound: 1757.0604318
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -438.1660156, 1804.7978516, -306.6542053, 1245.1453857, -1683.3111572, 2111.4519043
1: -535.2120361, 2015.2172852, -375.7231445, 1390.1154785, -1925.3275146, 2390.9404297
2: -621.6884155, 2042.3232422, -429.7876892, 1412.9300537, -2034.6182861, 2472.1108398
3: -876.6228027, 2243.8620605, -610.1465454, 1539.4003906, -2416.0231934, 2854.0085449
4: -1054.0126953, 2079.3713379, -721.1600952, 1436.7764893, -2490.7890625, 2800.5312500

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0854035, upper bound: 1757.1095417
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0854035, upper bound: 1757.1058974
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -441.2288513, 1821.4442139, -313.3518066, 1272.5496826, -1713.7785645, 2134.7958984
1: -539.5364380, 2033.7397461, -383.9295349, 1420.7796631, -1960.3159180, 2417.6689453
2: -626.0093384, 2061.7189941, -439.0760803, 1444.0350342, -2070.0441895, 2500.7946777
3: -884.4912720, 2262.6135254, -623.5477905, 1573.2559814, -2457.7468262, 2886.1613770
4: -1062.1268311, 2097.8671875, -736.9074707, 1468.3414307, -2530.4682617, 2834.7746582

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0371287, upper bound: 1757.1077246
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0384798, upper bound: 1757.1064151
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -438.1660156, 1804.7978516, -344.4634705, 1396.3786621, -1834.5445557, 2149.2612305
1: -535.2120361, 2015.2172852, -422.1711731, 1558.9559326, -2094.1679688, 2437.3884277
2: -621.6884155, 2042.3232422, -483.7297363, 1585.0070801, -2206.6955566, 2526.0524902
3: -876.6228027, 2243.8620605, -686.2062378, 1729.1444092, -2605.7670898, 2930.0683594
4: -1054.0126953, 2079.3713379, -812.8952026, 1612.3676758, -2666.3803711, 2892.2666016

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0888229, upper bound: 1757.1081977
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0888229, upper bound: 1757.1000565
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -441.2288513, 1821.4442139, -349.3248901, 1416.1810303, -1857.4099121, 2170.7690430
1: -539.5364380, 2033.7397461, -428.2104797, 1581.0932617, -2120.6296387, 2461.9499512
2: -626.0093384, 2061.7189941, -490.4802856, 1607.5646973, -2233.5737305, 2552.1992188
3: -884.4912720, 2262.6135254, -696.0726318, 1753.8930664, -2638.3842773, 2958.6860352
4: -1062.1268311, 2097.8671875, -824.4038696, 1635.3635254, -2697.4902344, 2922.2709961

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0322140, upper bound: 1757.0954015
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0385753, upper bound: 1757.0975195
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -378.1842957, 1554.5784912, -447.3091431, 1844.5843506, -2222.7685547, 2001.8876953
1: -462.3816833, 1735.4530029, -547.0040283, 2059.6230469, -2522.0046387, 2282.4567871
2: -536.3656006, 1760.1306152, -634.3399048, 2087.8959961, -2624.2614746, 2394.4704590
3: -756.5307007, 1930.6671143, -896.4486084, 2291.3088379, -3047.8395996, 2827.1142578
4: -908.3130493, 1790.7239990, -1075.8677979, 2124.4904785, -3032.8034668, 2866.5917969

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0261246, upper bound: 1757.0819047
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0237571, upper bound: 1757.0338508
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -385.2216797, 1589.5644531, -442.9012756, 1826.7816162, -2212.0034180, 2032.4654541
1: -470.5504150, 1774.0218506, -541.6278687, 2039.7738037, -2510.3239746, 2315.6496582
2: -547.8798828, 1799.2845459, -628.2280273, 2067.7236328, -2615.6035156, 2427.5124512
3: -771.5361328, 1976.5567627, -887.6599731, 2269.0839844, -3040.6201172, 2864.2163086
4: -930.1417847, 1831.4566650, -1065.4240723, 2103.9763184, -3034.1179199, 2896.8806152

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0247192, upper bound: 1757.0816052
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0223543, upper bound: 1757.0335292
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -418.2335815, 1726.7127686, -422.9389038, 1746.0081787, -2164.2412109, 2149.6516113
1: -511.6340027, 1928.8596191, -517.3671265, 1950.3825684, -2462.0163574, 2446.2265625
2: -592.1152344, 1955.3981934, -598.7854614, 1977.2226562, -2569.3378906, 2554.1835938
3: -837.6755371, 2141.4594727, -847.0921631, 2165.4328613, -3003.1083984, 2988.5515137
4: -1002.3061523, 1988.3162842, -1013.6734619, 2010.4298096, -3012.7358398, 3001.9897461

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0943404, upper bound: 1757.0934361
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0948670, upper bound: 1757.0965590
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -443.2770081, 1826.9315186, -422.9389038, 1746.0081787, -2189.2846680, 2249.8703613
1: -542.2296143, 2039.9180908, -517.3671265, 1950.3825684, -2492.6120605, 2557.2846680
2: -628.5104370, 2068.2482910, -598.7854614, 1977.2226562, -2605.7331543, 2667.0336914
3: -888.3466797, 2269.0156250, -847.0921631, 2165.4328613, -3053.7795410, 3116.1076660
4: -1065.5307617, 2104.4018555, -1013.6734619, 2010.4298096, -3075.9604492, 3118.0751953

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0943404, upper bound: 1757.0934361
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0948670, upper bound: 1757.0965590
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -436.1237183, 1796.1040039, -438.9853210, 1809.4025879, -2245.5263672, 2235.0893555
1: -532.7814941, 2005.5288086, -536.8937988, 2020.2792969, -2553.0607910, 2542.4221191
2: -618.7847900, 2032.5391846, -622.7307739, 2048.3686523, -2667.1530762, 2655.2700195
3: -872.5829468, 2233.0476074, -879.6583862, 2247.4042969, -3119.9865723, 3112.7053223
4: -1049.0150146, 2069.4274902, -1055.7158203, 2083.9948730, -3133.0097656, 3125.1433105

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0815998, upper bound: 1757.0262503
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0815998, upper bound: 1757.0378543
time: 0.75 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.98 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1145687, upper bound: 1757.1145687
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1145687, upper bound: 1757.1148268
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1148268, upper bound: 1757.1145687
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1148268, upper bound: 1757.1148268
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1146023, upper bound: 1757.1112526
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1146023, upper bound: 1757.1112526
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1148268, upper bound: 1757.1112526
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1148268, upper bound: 1757.1112526
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1112526, upper bound: 1757.1146023
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1112526, upper bound: 1757.1148268
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1112526, upper bound: 1757.1146023
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1112526, upper bound: 1757.1148268
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1112526, upper bound: 1757.1113094
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1112526, upper bound: 1757.1113094
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1112526, upper bound: 1757.1113094
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1112526, upper bound: 1757.1113094
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1077633, upper bound: 1757.0398151
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1029951, upper bound: 1757.0286552
IS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0360519, upper bound: 1757.0355836
IS_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0357210, upper bound: 1757.0343530
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1164592, upper bound: 1757.0852769
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1075388, upper bound: 1757.0382816
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1052588, upper bound: 1757.0852387
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1020657, upper bound: 1757.0382655
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1044493, upper bound: 1757.0935739
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1085965, upper bound: 1757.0966595
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1044493, upper bound: 1757.0935741
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1085965, upper bound: 1757.0966595
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1007618, upper bound: 1757.0896254
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1092399, upper bound: 1757.0954807
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1007618, upper bound: 1757.0896254
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1092402, upper bound: 1757.0954807
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0265817, upper bound: 1757.0895696
IS_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0248932, upper bound: 1757.0607556
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0251848, upper bound: 1757.0893851
IS_A2_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0234862, upper bound: 1757.0604318
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0854035, upper bound: 1757.1095417
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0854035, upper bound: 1757.1058974
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0371287, upper bound: 1757.1077246
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0384798, upper bound: 1757.1064151
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0888229, upper bound: 1757.1081977
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0888229, upper bound: 1757.1000565
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0322140, upper bound: 1757.0954015
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0385753, upper bound: 1757.0975195
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0261246, upper bound: 1757.0819047
IS_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0237571, upper bound: 1757.0338508
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0247192, upper bound: 1757.0816052
IS_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0223543, upper bound: 1757.0335292
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0943404, upper bound: 1757.0934361
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0948670, upper bound: 1757.0965590
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0943404, upper bound: 1757.0934361
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0948670, upper bound: 1757.0965590
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0815998, upper bound: 1757.0262503
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0815998, upper bound: 1757.0378543

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -272.6042480, 1111.4747314, -272.6042480, 1111.4747314, -1384.0788574, 1384.0788574
1: -334.2237854, 1241.4110107, -334.2237854, 1241.4110107, -1575.6347656, 1575.6347656
2: -381.0059814, 1261.2246094, -381.0059814, 1261.2246094, -1642.2304688, 1642.2304688
3: -542.2633057, 1371.0507812, -542.2633057, 1371.0507812, -1913.3140869, 1913.3140869
4: -637.6555786, 1281.7025146, -637.6555786, 1281.7025146, -1919.3580322, 1919.3580322

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1161966, upper bound: 1757.1124622
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1177641, upper bound: 1757.1147701
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -272.6042480, 1111.4747314, -311.7589417, 1265.6518555, -1538.2561035, 1423.2335205
1: -334.2237854, 1241.4110107, -381.9956665, 1413.0462646, -1747.2700195, 1623.4066162
2: -381.0059814, 1261.2246094, -436.8746033, 1436.2211914, -1817.2270508, 1698.0989990
3: -542.2633057, 1371.0507812, -620.3742065, 1564.9211426, -2107.1840820, 1991.4250488
4: -637.6555786, 1281.7025146, -733.2927246, 1460.4571533, -2098.1127930, 2014.9952393

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1161966, upper bound: 1757.1128858
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1177641, upper bound: 1757.1150537
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -311.7589417, 1265.6518555, -272.6042480, 1111.4747314, -1423.2335205, 1538.2561035
1: -381.9956665, 1413.0462646, -334.2237854, 1241.4110107, -1623.4066162, 1747.2700195
2: -436.8746033, 1436.2211914, -381.0059814, 1261.2246094, -1698.0989990, 1817.2270508
3: -620.3742065, 1564.9211426, -542.2633057, 1371.0507812, -1991.4250488, 2107.1838379
4: -733.2927246, 1460.4571533, -637.6555786, 1281.7025146, -2014.9952393, 2098.1127930

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0616061, upper bound: 1757.1113885
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1148268, upper bound: 1757.1145687
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -311.7589417, 1265.6518555, -311.7589417, 1265.6518555, -1577.4107666, 1577.4107666
1: -381.9956665, 1413.0462646, -381.9956665, 1413.0462646, -1795.0418701, 1795.0418701
2: -436.8746033, 1436.2211914, -436.8746033, 1436.2211914, -1873.0955811, 1873.0955811
3: -620.3742065, 1564.9211426, -620.3742065, 1564.9211426, -2185.2951660, 2185.2949219
4: -733.2927246, 1460.4571533, -733.2927246, 1460.4571533, -2193.7500000, 2193.7500000

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0616061, upper bound: 1757.1115870
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1148268, upper bound: 1757.1145760
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -272.6042480, 1111.4747314, -308.8294983, 1256.3496094, -1528.9538574, 1420.3040771
1: -334.2237854, 1241.4110107, -378.8326416, 1403.4211426, -1737.6448975, 1620.2435303
2: -381.0059814, 1261.2246094, -432.2922058, 1426.1948242, -1807.2008057, 1693.5168457
3: -542.2633057, 1371.0507812, -615.2606201, 1552.2734375, -2094.5366211, 1986.3112793
4: -637.6555786, 1281.7025146, -724.8505859, 1449.9963379, -2087.6518555, 2006.5529785

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1169309, upper bound: 1757.1035302
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1162323, upper bound: 1757.1089589
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1177978, upper bound: 1757.1114408
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -272.6042480, 1111.4747314, -349.6518555, 1416.9622803, -1689.5665283, 1461.1263428
1: -334.2237854, 1241.4110107, -428.5771790, 1581.9300537, -1916.1538086, 1669.9881592
2: -381.0059814, 1261.2246094, -490.9112549, 1608.4500732, -1989.4559326, 1752.1357422
3: -542.2633057, 1371.0507812, -696.6507568, 1754.8851318, -2297.1479492, 2067.7016602
4: -637.6555786, 1281.7025146, -825.1382446, 1636.2768555, -2273.9323730, 2106.8405762

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1169309, upper bound: 1757.1035302
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1162323, upper bound: 1757.1089589
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1177978, upper bound: 1757.1114408
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -311.7589417, 1265.6518555, -308.8294983, 1256.3496094, -1568.1085205, 1574.4813232
1: -381.9956665, 1413.0462646, -378.8326416, 1403.4211426, -1785.4167480, 1791.8787842
2: -436.8746033, 1436.2211914, -432.2922058, 1426.1948242, -1863.0693359, 1868.5134277
3: -620.3742065, 1564.9211426, -615.2606201, 1552.2734375, -2172.6477051, 2180.1813965
4: -733.2927246, 1460.4571533, -724.8505859, 1449.9963379, -2183.2888184, 2185.3076172

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1121398, upper bound: 1757.1033073
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0616177, upper bound: 1757.1081257
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1148268, upper bound: 1757.1112526
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -311.7589417, 1265.6518555, -349.6518555, 1416.9622803, -1728.7211914, 1615.3037109
1: -381.9956665, 1413.0462646, -428.5771790, 1581.9300537, -1963.9256592, 1841.6234131
2: -436.8746033, 1436.2211914, -490.9112549, 1608.4500732, -2045.3247070, 1927.1323242
3: -620.3742065, 1564.9211426, -696.6507568, 1754.8851318, -2375.2590332, 2261.5717773
4: -733.2927246, 1460.4571533, -825.1382446, 1636.2768555, -2369.5695801, 2285.5952148

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1121398, upper bound: 1757.1033073
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0616177, upper bound: 1757.1081257
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1148268, upper bound: 1757.1112526
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -309.5700378, 1259.4353027, -272.6042480, 1111.4747314, -1421.0445557, 1532.0395508
1: -379.7421265, 1406.8647461, -334.2237854, 1241.4110107, -1621.1530762, 1741.0885010
2: -433.3220215, 1429.6942139, -381.0059814, 1261.2246094, -1694.5466309, 1810.6999512
3: -616.7457275, 1556.0572510, -542.2633057, 1371.0507812, -1987.7963867, 2098.3203125
4: -726.5832520, 1453.5422363, -637.6555786, 1281.7025146, -2008.2857666, 2091.1977539

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1179714, upper bound: 1757.1149762
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1179918, upper bound: 1757.1149747
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -309.5700378, 1259.4353027, -311.7589417, 1265.6518555, -1575.2218018, 1571.1942139
1: -379.7421265, 1406.8647461, -381.9956665, 1413.0462646, -1792.7883301, 1788.8603516
2: -433.3220215, 1429.6942139, -436.8746033, 1436.2211914, -1869.5432129, 1866.5684814
3: -616.7457275, 1556.0572510, -620.3742065, 1564.9211426, -2181.6667480, 2176.4313965
4: -726.5832520, 1453.5422363, -733.2927246, 1460.4571533, -2187.0405273, 2186.8349609

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1085576, upper bound: 1757.1109840
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1095407, upper bound: 1757.1117892
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -349.7909546, 1417.5225830, -272.6042480, 1111.4747314, -1461.2653809, 1690.1268311
1: -428.7479553, 1582.5559082, -334.2237854, 1241.4110107, -1670.1589355, 1916.7796631
2: -491.1026001, 1609.0854492, -381.0059814, 1261.2246094, -1752.3270264, 1990.0911865
3: -696.9265137, 1755.5729980, -542.2633057, 1371.0507812, -2067.9772949, 2297.8361816
4: -825.4575806, 1636.9227295, -637.6555786, 1281.7025146, -2107.1599121, 2274.5781250

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0991402, upper bound: 1757.1102560
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1012832, upper bound: 1757.1113194
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -349.7909546, 1417.5225830, -311.7589417, 1265.6518555, -1615.4428711, 1729.2814941
1: -428.7479553, 1582.5559082, -381.9956665, 1413.0462646, -1841.7941895, 1964.5515137
2: -491.1026001, 1609.0854492, -436.8746033, 1436.2211914, -1927.3236084, 2045.9598389
3: -696.9265137, 1755.5729980, -620.3742065, 1564.9211426, -2261.8476562, 2375.9470215
4: -825.4575806, 1636.9227295, -733.2927246, 1460.4571533, -2285.9145508, 2370.2150879

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0991402, upper bound: 1757.1102560
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1012832, upper bound: 1757.1113566
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -309.5700378, 1259.4353027, -309.5700378, 1259.4353027, -1569.0051270, 1569.0051270
1: -379.7421265, 1406.8647461, -379.7421265, 1406.8647461, -1786.6069336, 1786.6069336
2: -433.3220215, 1429.6942139, -433.3220215, 1429.6942139, -1863.0161133, 1863.0161133
3: -616.7457275, 1556.0572510, -616.7457275, 1556.0572510, -2172.8029785, 2172.8029785
4: -726.5832520, 1453.5422363, -726.5832520, 1453.5422363, -2180.1252441, 2180.1252441

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1086150, upper bound: 1757.1017677
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1095407, upper bound: 1757.1017517
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -309.5700378, 1259.4353027, -349.7909546, 1417.5225830, -1727.0926514, 1609.2260742
1: -379.7421265, 1406.8647461, -428.7479553, 1582.5559082, -1962.2980957, 1835.6126709
2: -433.3220215, 1429.6942139, -491.1026001, 1609.0854492, -2042.4073486, 1920.7965088
3: -616.7457275, 1556.0572510, -696.9265137, 1755.5729980, -2372.3188477, 2252.9838867
4: -726.5832520, 1453.5422363, -825.4575806, 1636.9227295, -2363.5053711, 2278.9997559

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1086150, upper bound: 1757.1017677
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1095407, upper bound: 1757.1017517
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -349.7909546, 1417.5225830, -309.5700378, 1259.4353027, -1609.2260742, 1727.0926514
1: -428.7479553, 1582.5559082, -379.7421265, 1406.8647461, -1835.6126709, 1962.2980957
2: -491.1026001, 1609.0854492, -433.3220215, 1429.6942139, -1920.7965088, 2042.4074707
3: -696.9265137, 1755.5729980, -616.7457275, 1556.0572510, -2252.9838867, 2372.3188477
4: -825.4575806, 1636.9227295, -726.5832520, 1453.5422363, -2278.9997559, 2363.5056152

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0992291, upper bound: 1757.1013747
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1012832, upper bound: 1757.1013747
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -349.7909546, 1417.5225830, -349.7909546, 1417.5225830, -1767.3134766, 1767.3134766
1: -428.7479553, 1582.5559082, -428.7479553, 1582.5559082, -2011.3038330, 2011.3038330
2: -491.1026001, 1609.0854492, -491.1026001, 1609.0854492, -2100.1877441, 2100.1877441
3: -696.9265137, 1755.5729980, -696.9265137, 1755.5729980, -2452.4995117, 2452.4995117
4: -825.4575806, 1636.9227295, -825.4575806, 1636.9227295, -2462.3798828, 2462.3798828

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0992291, upper bound: 1757.1013747
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1012832, upper bound: 1757.1013747
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -308.4554443, 1251.6696777, -375.2522583, 1543.0726318, -1851.5280762, 1626.9217529
1: -378.0098267, 1397.4843750, -459.2923279, 1723.0449219, -2101.0546875, 1856.7767334
2: -432.0306396, 1420.3876953, -531.5819092, 1747.8581543, -2179.8886719, 1951.9692383
3: -613.7454224, 1547.3214111, -750.8627319, 1914.0628662, -2527.8083496, 2298.1840820
4: -724.8309326, 1444.4161377, -898.3192139, 1778.5295410, -2503.3603516, 2342.7353516

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1077451, upper bound: 1757.0398151
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1049680, upper bound: 1757.0383919
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -307.9263611, 1248.8166504, -391.8615723, 1607.4960938, -1915.4224854, 1640.6782227
1: -377.3312988, 1394.3511963, -479.5635986, 1794.4166260, -2171.7478027, 1873.9147949
2: -431.2268982, 1417.2352295, -554.0446167, 1822.3104248, -2253.5373535, 1971.2797852
3: -612.5296021, 1543.8038330, -783.1118774, 1993.6031494, -2606.1328125, 2326.9157715
4: -723.2389526, 1441.3375244, -935.7529907, 1853.4780273, -2576.7170410, 2377.0900879

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1027934, upper bound: 1757.0286552
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1029946, upper bound: 1757.0281211
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -304.3739319, 1235.1704102, -437.4574585, 1801.9121094, -2106.2861328, 1672.6276855
1: -372.9840698, 1378.9650879, -534.3391724, 2011.9718018, -2384.9550781, 1913.3041992
2: -426.4427490, 1401.6850586, -620.6896362, 2039.0390625, -2465.4819336, 2022.3747559
3: -605.5645752, 1526.9260254, -875.2121582, 2240.3261719, -2845.8906250, 2402.1381836
4: -715.2778931, 1425.4836426, -1052.3592529, 2076.0488281, -2791.3266602, 2477.8427734

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1089133, upper bound: 1757.0850933
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1053654, upper bound: 1757.0850933
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -311.0080566, 1262.2629395, -440.5417786, 1818.6151123, -2129.6230469, 1702.8046875
1: -381.1123657, 1409.2867432, -538.6918945, 2030.5695801, -2411.6818848, 1947.9786377
2: -435.6401978, 1432.4493408, -625.0411377, 2058.5019531, -2494.1420898, 2057.4904785
3: -618.8359375, 1560.4005127, -883.1232300, 2259.1557617, -2877.9912109, 2443.5236816
4: -730.8612671, 1456.7015381, -1060.5179443, 2094.6132812, -2825.4746094, 2517.2194824

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1075388, upper bound: 1757.0382816
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0994088, upper bound: 1757.0361230
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -304.0885010, 1234.8717041, -437.7729492, 1803.1719971, -2107.2604980, 1672.6446533
1: -372.5970459, 1378.6269531, -534.7345581, 2013.4045410, -2386.0014648, 1913.3615723
2: -426.1384277, 1401.2579346, -621.1370850, 2040.4844971, -2466.6228027, 2022.3950195
3: -605.0612793, 1526.5759277, -875.8369141, 2241.8395996, -2846.9008789, 2402.4128418
4: -714.9381714, 1424.9429932, -1053.0682373, 2077.5000000, -2792.4382324, 2478.0112305

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1010288, upper bound: 1757.0850617
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1015624, upper bound: 1757.0850617
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -310.7344971, 1262.0780029, -440.8311462, 1819.8161621, -2130.5507812, 1702.9091797
1: -380.7467346, 1409.0716553, -539.0505371, 2031.9197998, -2412.6662598, 1948.1221924
2: -435.3637085, 1432.1467285, -625.4541016, 2059.8698730, -2495.2333984, 2057.6008301
3: -618.3737793, 1560.1840820, -883.6961670, 2260.5888672, -2878.9626465, 2443.8801270
4: -730.5711060, 1456.2949219, -1061.1822510, 2095.9838867, -2826.5546875, 2517.4770508

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1020657, upper bound: 1757.0382655
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0964581, upper bound: 1757.0371265
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -308.7287903, 1255.7772217, -416.2709351, 1718.8725586, -2027.6010742, 1672.0480957
1: -378.7227478, 1402.7744141, -509.2720642, 1920.1009521, -2298.8229980, 1912.0465088
2: -432.1228027, 1425.5554199, -589.3342896, 1946.3798828, -2378.5026855, 2014.8896484
3: -615.0510254, 1551.5439453, -833.8949585, 2131.7753906, -2746.8264160, 2385.4389648
4: -724.5143433, 1449.3809814, -997.6843872, 1979.1986084, -2703.7128906, 2447.0654297

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1089504, upper bound: 1757.0936427
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1081440, upper bound: 1757.0936096
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -309.1094666, 1257.6446533, -420.9565735, 1737.7042236, -2046.8133545, 1678.6011963
1: -379.1797791, 1404.8612061, -514.9604492, 1941.0932617, -2320.2727051, 1919.8216553
2: -432.6823730, 1427.6575928, -595.9932251, 1967.8470459, -2400.5292969, 2023.6508789
3: -615.8455200, 1553.8350830, -843.1141357, 2155.0827637, -2770.9282227, 2396.9489746
4: -725.5218506, 1451.4666748, -1008.8666382, 2000.8907471, -2726.4125977, 2460.3332520

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1109513, upper bound: 1757.0967324
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1087727, upper bound: 1757.0966946
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -349.0191040, 1414.2442627, -416.2709351, 1718.8725586, -2067.8916016, 1830.5151367
1: -427.8117371, 1578.8859863, -509.2720642, 1920.1009521, -2347.9125977, 2088.1577148
2: -490.0200500, 1605.3760986, -589.3342896, 1946.3798828, -2436.3999023, 2194.7104492
3: -695.3892212, 1751.5506592, -833.8949585, 2131.7753906, -2827.1645508, 2585.4455566
4: -823.6232300, 1633.1781006, -997.6843872, 1979.1986084, -2802.8217773, 2630.8625488

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1043850, upper bound: 1757.0934349
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1019990, upper bound: 1757.0933411
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -349.2679443, 1415.4268799, -420.9565735, 1737.7042236, -2086.9721680, 1836.3834229
1: -428.1112061, 1580.2139893, -514.9604492, 1941.0932617, -2369.2041016, 2095.1738281
2: -490.3604431, 1606.7064209, -595.9932251, 1967.8470459, -2458.2075195, 2202.6997070
3: -695.8930054, 1752.9587402, -843.1141357, 2155.0827637, -2850.9758301, 2596.0727539
4: -824.1989746, 1634.5126953, -1008.8666382, 2000.8907471, -2825.0898438, 2643.3793945

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1084456, upper bound: 1757.0965710
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1026331, upper bound: 1757.0963980
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -304.2746277, 1237.0314941, -442.1720276, 1821.6081543, -2125.8828125, 1679.2032471
1: -373.2769165, 1381.8488770, -540.9208984, 2033.8192139, -2407.0957031, 1922.7696533
2: -425.8387146, 1404.3037109, -626.4272461, 2063.1657715, -2489.0041504, 2030.7309570
3: -606.1372681, 1528.3574219, -886.0424194, 2261.7121582, -2867.8491211, 2414.3994141
4: -713.9691162, 1427.7504883, -1061.7980957, 2098.3588867, -2812.3281250, 2489.5485840

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1063582, upper bound: 1757.0893061
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1064403, upper bound: 1757.0893140
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -308.8629761, 1256.5830078, -441.1957397, 1818.2915039, -2127.1545410, 1697.7785645
1: -378.8821411, 1403.6749268, -539.6814575, 2030.2332764, -2409.1147461, 1943.3562012
2: -432.3293762, 1426.4597168, -625.5986328, 2058.5107422, -2490.8400879, 2052.0583496
3: -615.3497314, 1552.5187988, -884.2444458, 2258.3129883, -2873.6625977, 2436.7631836
4: -724.9079590, 1450.2586670, -1060.7050781, 2094.2919922, -2819.1994629, 2510.9638672

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0946274, upper bound: 1757.0384817
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0954015, upper bound: 1757.0322140
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -344.8559570, 1396.6663818, -442.1720276, 1821.6081543, -2166.4641113, 1838.8382568
1: -422.6806641, 1559.2441406, -540.9208984, 2033.8192139, -2456.4997559, 2100.1650391
2: -484.1601562, 1585.4267578, -626.4272461, 2063.1657715, -2547.3259277, 2211.8540039
3: -687.0415039, 1730.0507812, -886.0424194, 2261.7121582, -2948.7531738, 2616.0932617
4: -813.8773804, 1612.8668213, -1061.7980957, 2098.3588867, -2912.2363281, 2674.6650391

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0918556, upper bound: 1757.0876244
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1007618, upper bound: 1757.0895217
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -349.0173645, 1414.3367920, -441.1957397, 1818.2915039, -2167.3088379, 1855.5322266
1: -427.8110962, 1578.9989014, -539.6814575, 2030.2332764, -2458.0439453, 2118.6799316
2: -490.0027466, 1605.4813232, -625.5986328, 2058.5107422, -2548.5134277, 2231.0798340
3: -695.3883667, 1751.6109619, -884.2444458, 2258.3129883, -2953.7014160, 2635.8554688
4: -823.5729370, 1633.2808838, -1060.7050781, 2094.2919922, -2917.8649902, 2693.9858398

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0361179, upper bound: 1757.0349694
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0397367, upper bound: 1757.0349694
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -371.0222168, 1524.4816895, -320.3376160, 1296.2031250, -1667.2252197, 1844.8192139
1: -453.7417603, 1701.9765625, -392.2883301, 1447.5819092, -1901.3234863, 2094.2644043
2: -526.2329102, 1726.2471924, -451.3689575, 1471.1108398, -1997.3436279, 2177.6162109
3: -742.1495972, 1893.2103271, -637.3461304, 1608.5780029, -2350.7272949, 2530.5556641
4: -890.7204590, 1756.5201416, -758.0823975, 1498.4545898, -2389.1750488, 2514.6022949

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0262392, upper bound: 1757.0860270
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0186717, upper bound: 1757.0889834
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0123810, upper bound: 1757.0770518
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0265817, upper bound: 1757.0895696
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 21

Time for candidate selection: 11.28 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0263465, upper bound: 1757.0890090
time: 0.96 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0246891, upper bound: 1757.0893201
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0258182, upper bound: 1757.0894377
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -378.4457092, 1560.8878174, -315.0918274, 1274.8421631, -1653.2878418, 1875.9796143
1: -462.3668213, 1742.0853271, -385.8719788, 1423.6687012, -1886.0354004, 2127.9570312
2: -538.2493286, 1767.0178223, -444.0158997, 1446.8010254, -1985.0501709, 2211.0334473
3: -757.8331299, 1940.7590332, -626.8779297, 1581.8576660, -2339.6906738, 2567.6367188
4: -913.3524170, 1798.7465820, -745.5827026, 1473.6710205, -2387.0234375, 2544.3288574

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0248431, upper bound: 1757.0858562
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0110393, upper bound: 1757.0768425
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0251848, upper bound: 1757.0893851
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 49

Time for candidate selection: 10.91 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0249559, upper bound: 1757.0887403
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0231043, upper bound: 1757.0891360
time: 1.04 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0244214, upper bound: 1757.0892628
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -435.0601501, 1792.4847412, -265.4828796, 1082.6499023, -1517.7098389, 2057.9672852
1: -531.3702393, 2001.4381104, -325.5089111, 1209.1358643, -1740.5061035, 2326.9465332
2: -617.3857422, 2028.2801514, -371.0965576, 1228.5264893, -1845.9122314, 2399.3762207
3: -870.4490356, 2228.6887207, -528.1140747, 1335.3281250, -2205.7770996, 2756.8027344
4: -1046.9327393, 2065.0812988, -620.8587646, 1248.4377441, -2295.3706055, 2685.9399414

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0452408, upper bound: 1757.1074911
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0831593, upper bound: 1757.1095417
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -436.1237183, 1796.1040039, -303.7132874, 1233.0361328, -1669.1597900, 2099.8173828
1: -532.7814941, 2005.5288086, -372.1496277, 1376.5568848, -1909.3383789, 2377.6774902
2: -618.7847900, 2032.5391846, -425.6797180, 1399.2200928, -2018.0047607, 2458.2189941
3: -872.5829468, 2233.0476074, -604.3227539, 1524.4552002, -2397.0373535, 2837.3703613
4: -1049.0150146, 2069.4274902, -714.2428589, 1422.8752441, -2471.8901367, 2783.6704102

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0451534, upper bound: 1757.1058974
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0831593, upper bound: 1757.1055735
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -439.6908569, 1815.3186035, -313.9205627, 1276.5225830, -1716.2133789, 2129.2390137
1: -537.6618652, 2026.8853760, -384.6399231, 1424.9396973, -1962.6015625, 2411.5253906
2: -623.8649292, 2054.7841797, -440.1931763, 1448.4901123, -2072.3549805, 2494.9772949
3: -881.4392700, 2254.9919434, -625.0411377, 1578.1131592, -2459.5524902, 2880.0332031
4: -1058.5231934, 2090.8173828, -739.1383057, 1472.7161865, -2531.2392578, 2829.9550781

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0287323, upper bound: 1757.1077246
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0205149, upper bound: 1757.0988796
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0371287, upper bound: 1757.1077246
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0340375, upper bound: 1757.1056442
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0340375, upper bound: 1757.1064151
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -440.6536255, 1819.0991211, -310.8813477, 1262.5406494, -1703.1942139, 2129.9804688
1: -538.8317261, 2031.1182861, -380.9079895, 1409.6031494, -1948.4348145, 2412.0261230
2: -625.2076416, 2059.0644531, -435.6134644, 1432.7138672, -2057.9211426, 2494.6779785
3: -883.3366699, 2259.7094727, -618.5988159, 1560.8190918, -2444.1555176, 2878.3081055
4: -1060.7700195, 2095.1657715, -731.0195923, 1456.8098145, -2517.5798340, 2826.1853027

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0300180, upper bound: 1757.1064151
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0219804, upper bound: 1757.0973065
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0384798, upper bound: 1757.1064151
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0352863, upper bound: 1757.1056442
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0352863, upper bound: 1757.1064151
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -435.0601501, 1792.4847412, -301.6807861, 1227.7254639, -1662.7855225, 2094.1655273
1: -531.3702393, 2001.4381104, -370.0173035, 1371.3906250, -1902.7608643, 2371.4553223
2: -617.3857422, 2028.2801514, -422.3810730, 1393.6711426, -2011.0568848, 2450.6608887
3: -870.4490356, 2228.6887207, -600.9207153, 1516.3566895, -2386.8056641, 2829.6093750
4: -1046.9327393, 2065.0812988, -708.0236206, 1416.7507324, -2463.6835938, 2773.1049805

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0453571, upper bound: 1757.1032340
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0886727, upper bound: 1757.1081977
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -436.1237183, 1796.1040039, -341.5024109, 1384.0561523, -1820.1798096, 2137.6064453
1: -532.7814941, 2005.5288086, -418.5633850, 1545.1550293, -2077.9360352, 2424.0915527
2: -618.7847900, 2032.5391846, -479.5695496, 1571.0903320, -2189.8750000, 2512.1086426
3: -872.5829468, 2233.0476074, -680.2988281, 1713.9654541, -2586.5476074, 2913.3464355
4: -1049.0150146, 2069.4274902, -805.8392944, 1598.2606201, -2647.2756348, 2875.2668457

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0452414, upper bound: 1757.0990446
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0886727, upper bound: 1757.1000563
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -438.1390686, 1809.1595459, -305.9045410, 1244.5888672, -1682.7279053, 2115.0634766
1: -535.7105103, 2020.0081787, -375.2692566, 1390.2646484, -1925.9750977, 2395.2768555
2: -621.7402954, 2047.7425537, -428.1958923, 1412.9000244, -2034.6403809, 2475.9384766
3: -878.3416748, 2247.5737305, -609.4616089, 1537.7893066, -2416.1308594, 2857.0354004
4: -1055.0895996, 2083.6335449, -717.9440918, 1436.6033936, -2491.6918945, 2801.5769043

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0232789, upper bound: 1757.0954015
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=2002.608154296875
rel_dist={0: [-1757.1264512206008, 1757.1264512206008]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1163285, upper bound: 1757.1055601
time: 0.80 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1045510, upper bound: 1757.1045510
time: 0.69 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.60 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.60
Output dim: 0, lower bound: -1757.1163285, upper bound: 1757.1055601
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.60
Output dim: 0, lower bound: -1757.1045510, upper bound: 1757.1045510

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -356.5766602, 1445.0728760, -373.7850037, 1516.3778076, -1872.9544678, 1818.8579102
1: -437.0072937, 1613.3231201, -458.0685730, 1693.2730713, -2130.2800293, 2071.3916016
2: -500.5809937, 1640.3117676, -524.4805908, 1721.1791992, -2221.7600098, 2164.7922363
3: -710.3735962, 1789.6778564, -744.5432739, 1876.7159424, -2587.0893555, 2534.2211914
4: -841.4490967, 1668.5886230, -881.2928467, 1750.6546631, -2592.1037598, 2549.8813477

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1044713, upper bound: 1757.1044713
time: 0.65 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1044713, upper bound: 1757.1045505
time: 0.68 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -457.7469177, 1887.2629395, -372.7506714, 1519.4101562, -1977.1571045, 2260.0134277
1: -559.7200317, 2107.2399902, -456.7657166, 1696.7579346, -2256.4780273, 2564.0056152
2: -649.0897827, 2136.3076172, -523.5133057, 1724.5603027, -2373.6501465, 2659.8203125
3: -917.2413330, 2344.1276855, -742.9725342, 1878.8721924, -2796.1135254, 3087.1000977
4: -1100.8392334, 2173.5678711, -880.0524902, 1753.6262207, -2854.4653320, 3053.6203613

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0672346, upper bound: 1757.0377397
time: 0.90 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1045510, upper bound: 1757.1045510
time: 0.76 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.00 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.00
Output dim: 0, lower bound: -1757.1044713, upper bound: 1757.1044713
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.00
Output dim: 0, lower bound: -1757.1044713, upper bound: 1757.1045505
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 3.00
Output dim: 0, lower bound: -1757.0672346, upper bound: 1757.0377397
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.00
Output dim: 0, lower bound: -1757.1045510, upper bound: 1757.1045510

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -356.5766602, 1445.0728760, -356.5766602, 1445.0728760, -1801.6495361, 1801.6495361
1: -437.0072937, 1613.3231201, -437.0072937, 1613.3231201, -2050.3303223, 2050.3303223
2: -500.5809937, 1640.3117676, -500.5809937, 1640.3117676, -2140.8925781, 2140.8928223
3: -710.3735962, 1789.6778564, -710.3735962, 1789.6778564, -2500.0515137, 2500.0515137
4: -841.4490967, 1668.5886230, -841.4490967, 1668.5886230, -2510.0375977, 2510.0375977

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1025650, upper bound: 1757.0697894
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1163285, upper bound: 1757.1055423
time: 0.70 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -356.5766602, 1445.0728760, -457.6540527, 1886.8894043, -2243.4658203, 1902.7269287
1: -437.0072937, 1613.3231201, -559.6082153, 2106.8259277, -2543.8332520, 2172.9311523
2: -500.5809937, 1640.3117676, -648.9575806, 2135.8837891, -2636.4645996, 2289.2690430
3: -710.3735962, 1789.6778564, -917.0573730, 2343.6613770, -3054.0349121, 2706.7351074
4: -841.4490967, 1668.5886230, -1100.6107178, 2173.1391602, -3014.5883789, 2769.1992188

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1025650, upper bound: 1757.0697894
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1163285, upper bound: 1757.1055601
time: 0.74 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -453.2650146, 1869.0682373, -370.3179321, 1509.7926025, -1963.0576172, 2239.3862305
1: -554.2871704, 2086.9489746, -453.8044128, 1686.0285645, -2240.3154297, 2540.7534180
2: -642.7421265, 2115.7058105, -520.1462402, 1713.6300049, -2356.3720703, 2635.8515625
3: -908.3111572, 2321.4531250, -738.1925659, 1866.9652100, -2775.2763672, 3059.6457520
4: -1089.9895020, 2152.6918945, -874.4108276, 1742.5361328, -2832.5256348, 3027.1027832

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0966250, upper bound: 1757.0947514
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0947514, upper bound: 1757.0947514
time: 0.89 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.99 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 0, lower bound: -1757.1025650, upper bound: 1757.0697894
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 0, lower bound: -1757.1163285, upper bound: 1757.1055423
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 0, lower bound: -1757.1025650, upper bound: 1757.0697894
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 0, lower bound: -1757.1163285, upper bound: 1757.1055601
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 0, lower bound: -1757.0966250, upper bound: 1757.0947514
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 0, lower bound: -1757.0947514, upper bound: 1757.0947514

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -314.7524109, 1277.9980469, -341.4608154, 1382.8364258, -1697.5887451, 1619.4587402
1: -385.6337891, 1426.8695068, -418.6041870, 1543.8013916, -1929.4350586, 1845.4736328
2: -441.0567932, 1450.1931152, -479.1699829, 1569.8137207, -2010.8704834, 1929.3630371
3: -626.3041992, 1580.1624756, -680.1207275, 1712.8166504, -2339.1203613, 2260.2829590
4: -740.3444824, 1474.6290283, -805.1219482, 1597.2030029, -2337.5473633, 2279.7507324

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1137332, upper bound: 1757.1172965
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1133992, upper bound: 1757.1107440
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -352.7528076, 1429.8717041, -354.2008667, 1435.6182861, -1788.3710938, 1784.0725098
1: -432.3590393, 1596.3819580, -434.1175537, 1602.7880859, -2035.1470947, 2030.4995117
2: -495.2670288, 1623.0344238, -497.2729492, 1629.5616455, -2124.8286133, 2120.3073730
3: -702.8384399, 1770.8116455, -705.6884155, 1777.9489746, -2480.7871094, 2476.4997559
4: -832.5136719, 1651.0700684, -835.8913574, 1657.6831055, -2490.1967773, 2486.9614258

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1111047, upper bound: 1757.1173490
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1107725, upper bound: 1757.1107725
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -314.7524109, 1277.9980469, -445.0621948, 1835.8236084, -2150.5759277, 1723.0601807
1: -385.6337891, 1426.8695068, -544.0747681, 2049.7810059, -2435.4145508, 1970.9443359
2: -441.0567932, 1450.1931152, -631.3070068, 2077.8967285, -2518.9536133, 2081.5000000
3: -626.3041992, 1580.1624756, -891.9097900, 2280.9343262, -2907.2382812, 2472.0717773
4: -740.3444824, 1474.6290283, -1071.2692871, 2114.3474121, -2854.6918945, 2545.8977051

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0919372, upper bound: 1757.0582811
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0795080, upper bound: 1757.0365321
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -352.7528076, 1429.8717041, -453.1716919, 1868.6922607, -2221.4450684, 1883.0430908
1: -432.3590393, 1596.3819580, -554.1748047, 2086.5322266, -2518.8911133, 2150.5566406
2: -495.2670288, 1623.0344238, -642.6094360, 2115.2792969, -2610.5463867, 2265.6437988
3: -702.8384399, 1770.8116455, -908.1264038, 2320.9841309, -3023.8225098, 2678.9377441
4: -832.5136719, 1651.0700684, -1089.7593994, 2152.2607422, -2984.7744141, 2740.8295898

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1031515, upper bound: 1757.0965722
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1072348, upper bound: 1757.0947950
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -421.0489197, 1738.2698975, -361.4754333, 1474.1966553, -1895.2456055, 2099.7451172
1: -515.0668335, 1941.7468262, -443.0010071, 1646.2828369, -2161.3496094, 2384.7473145
2: -596.1104126, 1968.4721680, -507.8663940, 1673.2893066, -2269.3996582, 2476.3381348
3: -843.3159790, 2155.8198242, -720.7009277, 1822.9195557, -2666.2353516, 2876.5207520
4: -1009.1129761, 2001.5656738, -853.7929077, 1701.5068359, -2710.6193848, 2855.3581543

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0964088, upper bound: 1757.0946528
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0964088, upper bound: 1757.0947514
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -446.3280029, 1839.3863525, -368.4134216, 1501.8244629, -1948.1524658, 2207.7995605
1: -545.9478149, 2053.8002930, -451.4934998, 1677.1024170, -2223.0500488, 2505.2932129
2: -632.8404541, 2082.3535156, -517.4544067, 1704.6738281, -2337.5141602, 2599.8078613
3: -894.4564819, 2284.5166016, -734.3791504, 1857.1232910, -2751.5795898, 3018.8955078
4: -1072.9165039, 2118.6967773, -869.8020020, 1733.4395752, -2806.3559570, 2988.4985352

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0945347, upper bound: 1757.0946527
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0945347, upper bound: 1757.0947514
time: 0.70 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.90 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -1757.1137332, upper bound: 1757.1172965
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -1757.1133992, upper bound: 1757.1107440
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -1757.1111047, upper bound: 1757.1173490
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -1757.1107725, upper bound: 1757.1107725
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -1757.0919372, upper bound: 1757.0582811
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -1757.0795080, upper bound: 1757.0365321
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -1757.1031515, upper bound: 1757.0965722
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -1757.1072348, upper bound: 1757.0947950
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -1757.0964088, upper bound: 1757.0946528
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -1757.0964088, upper bound: 1757.0947514
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -1757.0945347, upper bound: 1757.0946527
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -1757.0945347, upper bound: 1757.0947514

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -306.0153198, 1242.8343506, -297.6805115, 1209.8760986, -1515.8913574, 1540.5147705
1: -374.9665222, 1387.6201172, -365.2685852, 1351.4611816, -1726.4277344, 1752.8886719
2: -429.0001526, 1410.2749023, -416.4762573, 1373.5759277, -1802.5760498, 1826.7512207
3: -609.0453491, 1536.8415527, -592.8800659, 1495.0766602, -2104.1220703, 2129.7214355
4: -720.3540039, 1433.9733887, -697.8868408, 1397.0240479, -2117.3776855, 2131.8596191

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1133992, upper bound: 1757.1107440
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1133992, upper bound: 1757.1107440
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -313.0661926, 1271.0329590, -338.6444702, 1371.1126709, -1684.1788330, 1609.6772461
1: -383.5854187, 1419.0708008, -415.1849670, 1530.6711426, -1914.2565918, 1834.2556152
2: -438.7011414, 1442.3132324, -475.2209167, 1556.5743408, -1995.2752686, 1917.5338135
3: -622.9619751, 1571.5617676, -674.5199585, 1698.2709961, -2321.2326660, 2246.0815430
4: -736.3681641, 1466.6364746, -798.4213257, 1583.7113037, -2320.0795898, 2265.0576172

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0590093, upper bound: 1757.0919675
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1133992, upper bound: 1757.1107440
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -343.4269104, 1392.4060059, -310.8717651, 1264.5526123, -1607.9794922, 1703.2778320
1: -420.9422302, 1554.5615234, -381.3268127, 1412.5860596, -1833.5283203, 1935.8883057
2: -482.2944641, 1580.5328369, -435.1131592, 1435.5080566, -1917.8024902, 2015.6458740
3: -684.3370972, 1724.6115723, -619.2958374, 1562.4045410, -2246.7414551, 2343.9072266
4: -810.8817139, 1607.7896729, -729.5848389, 1459.4410400, -2270.3227539, 2337.3745117

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1091949, upper bound: 1757.1113645
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1030042, upper bound: 1757.1109154
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -351.0979919, 1422.9626465, -351.2284546, 1423.2247314, -1774.3226318, 1774.1910400
1: -430.3428040, 1588.6464844, -430.4937744, 1588.9121094, -2019.2545166, 2019.1402588
2: -492.9394531, 1615.2322998, -493.0940247, 1615.5609131, -2108.5000000, 2108.3261719
3: -699.5347290, 1762.2705078, -699.7558594, 1762.6485596, -2462.1833496, 2462.0263672
4: -828.5653076, 1643.1577148, -828.8113403, 1643.4842529, -2472.0495605, 2471.9685059

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1083502, upper bound: 1757.1027868
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1025262, upper bound: 1757.1025262
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -300.8891907, 1222.0455322, -432.1781006, 1781.2415771, -2082.1308594, 1654.2236328
1: -368.6730347, 1364.2672119, -527.6552124, 1988.6503906, -2357.3229980, 1891.9223633
2: -421.8049927, 1386.6715088, -613.5440063, 2015.3350830, -2437.1401367, 2000.2153320
3: -598.7077026, 1510.7580566, -864.7363281, 2215.3664551, -2814.0739746, 2375.4943848
4: -707.6121826, 1410.1047363, -1040.9542236, 2051.9641113, -2759.5761719, 2451.0590820

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0440028, upper bound: 1757.0501630
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0909811, upper bound: 1757.0510305
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -312.3126221, 1268.5493164, -437.4534302, 1806.0596924, -2118.3723145, 1706.0024414
1: -382.6657104, 1416.3085938, -534.7276611, 2016.4947510, -2399.1604004, 1951.0361328
2: -437.6087952, 1439.5113525, -620.7834473, 2044.1401367, -2481.7490234, 2060.2944336
3: -621.5132446, 1568.1770020, -876.9152222, 2244.1667480, -2865.6799316, 2445.0922852
4: -734.3678589, 1463.7189941, -1053.8702393, 2080.0532227, -2814.4211426, 2517.5891113

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0723485, upper bound: 1757.0357739
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0795080, upper bound: 1757.0365321
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -343.4269104, 1392.4060059, -420.3494873, 1735.5183105, -2078.9453125, 1812.7554932
1: -420.9422302, 1554.5615234, -514.2151489, 1938.6901855, -2359.6323242, 2068.7766113
2: -482.2944641, 1580.5328369, -595.1538086, 1965.3167725, -2447.6105957, 2175.6865234
3: -684.3370972, 1724.6115723, -841.9440918, 2152.4304199, -2836.7670898, 2566.5554199
4: -810.8817139, 1607.7896729, -1007.4937134, 1998.3969727, -2809.2788086, 2615.2834473

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1031250, upper bound: 1757.0954890
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1022828, upper bound: 1757.0964740
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -351.0979919, 1422.9626465, -446.2417908, 1839.0377197, -2190.1357422, 1869.2044678
1: -430.3428040, 1588.6464844, -545.8438110, 2053.4138184, -2483.7565918, 2134.4902344
2: -492.9394531, 1615.2322998, -632.7182617, 2081.9565430, -2574.8957520, 2247.9506836
3: -699.5347290, 1762.2705078, -894.2856445, 2284.0830078, -2983.6176758, 2656.5561523
4: -828.5653076, 1643.1577148, -1072.7053223, 2118.2973633, -2946.8627930, 2715.8625488

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1066581, upper bound: 1757.0947061
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1022828, upper bound: 1757.0947029
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -421.0489197, 1738.2698975, -343.4269104, 1392.4060059, -1813.4549561, 2081.6967773
1: -515.0668335, 1941.7468262, -420.9422302, 1554.5615234, -2069.6284180, 2362.6889648
2: -596.1104126, 1968.4721680, -482.2944641, 1580.5328369, -2176.6433105, 2450.7658691
3: -843.3159790, 2155.8198242, -684.3370972, 1724.6115723, -2567.9270020, 2840.1564941
4: -1009.1129761, 2001.5656738, -810.8817139, 1607.7896729, -2616.9025879, 2812.4470215

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0926368, upper bound: 1757.0938057
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0956388, upper bound: 1757.0944006
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -421.0489197, 1738.2698975, -444.6355896, 1834.4644775, -2255.5134277, 2182.9055176
1: -515.0668335, 1941.7468262, -543.6809082, 2048.3271484, -2563.3940430, 2485.4277344
2: -596.1104126, 1968.4721680, -630.6976318, 2076.4060059, -2672.5163574, 2599.1691895
3: -843.3159790, 2155.8198242, -891.1262817, 2278.7812500, -3122.0971680, 3046.9460449
4: -1009.1129761, 2001.5656738, -1069.8513184, 2112.7680664, -3121.8811035, 3071.4165039

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0926368, upper bound: 1757.0938145
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0956388, upper bound: 1757.0945201
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -446.3280029, 1839.3863525, -351.0979919, 1422.9626465, -1869.2906494, 2190.4843750
1: -545.9478149, 2053.8002930, -430.3428040, 1588.6464844, -2134.5942383, 2484.1430664
2: -632.8404541, 2082.3535156, -492.9394531, 1615.2322998, -2248.0727539, 2575.2929688
3: -894.4564819, 2284.5166016, -699.5347290, 1762.2705078, -2656.7268066, 2984.0512695
4: -1072.9165039, 2118.6967773, -828.5653076, 1643.1577148, -2716.0737305, 2947.2622070

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0684062, upper bound: 1757.0362734
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0343889, upper bound: 1757.0343889
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -446.3280029, 1839.3863525, -446.2483521, 1839.7030029, -2286.0307617, 2285.6347656
1: -545.9478149, 2053.8002930, -545.8134155, 2054.1838379, -2600.1315918, 2599.6132812
2: -632.8404541, 2082.3535156, -632.7642212, 2082.5539551, -2715.3945312, 2715.1176758
3: -894.4564819, 2284.5166016, -894.3014526, 2284.9060059, -3179.3625488, 3178.8178711
4: -1072.9165039, 2118.6967773, -1072.8637695, 2119.0290527, -3191.9453125, 3191.5600586

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0684062, upper bound: 1757.0362734
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0343889, upper bound: 1757.0343889
time: 0.64 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.99 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -1757.1133992, upper bound: 1757.1107440
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -1757.1133992, upper bound: 1757.1107440
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -1757.0590093, upper bound: 1757.0919675
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -1757.1133992, upper bound: 1757.1107440
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -1757.1091949, upper bound: 1757.1113645
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -1757.1030042, upper bound: 1757.1109154
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -1757.1083502, upper bound: 1757.1027868
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -1757.1025262, upper bound: 1757.1025262
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.99
Output dim: 0, lower bound: -1757.0440028, upper bound: 1757.0501630
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -1757.0909811, upper bound: 1757.0510305
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.99
Output dim: 0, lower bound: -1757.0723485, upper bound: 1757.0357739
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -1757.0795080, upper bound: 1757.0365321
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -1757.1031250, upper bound: 1757.0954890
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -1757.1022828, upper bound: 1757.0964740
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -1757.1066581, upper bound: 1757.0947061
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -1757.1022828, upper bound: 1757.0947029
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -1757.0926368, upper bound: 1757.0938057
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -1757.0956388, upper bound: 1757.0944006
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -1757.0926368, upper bound: 1757.0938145
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -1757.0956388, upper bound: 1757.0945201
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.99
Output dim: 0, lower bound: -1757.0684062, upper bound: 1757.0362734
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.99
Output dim: 0, lower bound: -1757.0343889, upper bound: 1757.0343889
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.99
Output dim: 0, lower bound: -1757.0684062, upper bound: 1757.0362734
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.99
Output dim: 0, lower bound: -1757.0343889, upper bound: 1757.0343889

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -272.6042480, 1111.4747314, -297.6805115, 1209.8760986, -1482.4803467, 1409.1550293
1: -334.2237854, 1241.4110107, -365.2685852, 1351.4611816, -1685.6849365, 1606.6795654
2: -381.0059814, 1261.2246094, -416.4762573, 1373.5759277, -1754.5819092, 1677.7009277
3: -542.2633057, 1371.0507812, -592.8800659, 1495.0766602, -2037.3399658, 1963.9305420
4: -637.6555786, 1281.7025146, -697.8868408, 1397.0240479, -2034.6794434, 1979.5893555

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1137332, upper bound: 1757.1170720
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1137332, upper bound: 1757.1172965
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -311.7570190, 1265.6446533, -297.6805115, 1209.8760986, -1521.6330566, 1563.3250732
1: -381.9933777, 1413.0380859, -365.2685852, 1351.4611816, -1733.4543457, 1778.3066406
2: -436.8719788, 1436.2128906, -416.4762573, 1373.5759277, -1810.4477539, 1852.6892090
3: -620.3704224, 1564.9118652, -592.8800659, 1495.0766602, -2115.4470215, 2157.7917480
4: -733.2883911, 1460.4486084, -697.8868408, 1397.0240479, -2130.3120117, 2158.3349609

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1137332, upper bound: 1757.1170720
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1137332, upper bound: 1757.1172965
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -306.9274597, 1237.4173584, -329.5258179, 1334.7794189, -1641.7069092, 1566.9431152
1: -375.7303467, 1382.0565186, -404.1005554, 1490.2194824, -1865.9498291, 1786.1569824
2: -429.6966553, 1404.1936035, -462.1606445, 1515.3928223, -1945.0894775, 1866.3541260
3: -609.3472900, 1534.1628418, -656.3840942, 1652.4611816, -2261.8085938, 2190.5468750
4: -721.7412109, 1429.1094971, -775.9920654, 1541.8696289, -2263.6103516, 2205.1010742

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0590093, upper bound: 1757.0919675
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0590092, upper bound: 1757.0919675
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -308.8428345, 1253.4255371, -336.7215271, 1363.1037598, -1671.9465332, 1590.1470947
1: -378.4500122, 1399.4350586, -412.8457642, 1521.7381592, -1900.1881104, 1812.2805176
2: -432.7960205, 1422.3830566, -472.5453796, 1547.5144043, -1980.3103027, 1894.9284668
3: -614.5263062, 1549.8706055, -670.6832886, 1688.4141846, -2302.9404297, 2220.5539551
4: -726.3590698, 1446.4643555, -793.9003906, 1574.5244141, -2300.8835449, 2240.3645020

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1133992, upper bound: 1757.1107440
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1133992, upper bound: 1757.1107440
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -334.6288452, 1358.0841064, -307.0610962, 1249.7287598, -1584.3576660, 1665.1452637
1: -410.2295837, 1516.1625977, -376.6734314, 1395.9779053, -1806.2072754, 1892.8359375
2: -470.1205139, 1541.5301514, -429.8460083, 1418.6618652, -1888.7821045, 1971.3760986
3: -667.1864624, 1682.5451660, -611.8773804, 1544.3198242, -2211.5063477, 2294.4223633
4: -790.7907715, 1568.3714600, -720.9025269, 1442.5714111, -2233.3623047, 2289.2739258

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1030042, upper bound: 1757.1109154
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1030042, upper bound: 1757.1109153
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -365.0411987, 1479.7664795, -308.2574463, 1253.8942871, -1618.9350586, 1788.0239258
1: -447.6459656, 1652.1899414, -378.1238403, 1400.6822510, -1848.3281250, 2030.3137207
2: -512.7231445, 1679.9470215, -431.4377136, 1423.4355469, -1936.1586914, 2111.3847656
3: -727.6420898, 1833.4910889, -614.0556641, 1548.8579102, -2276.5000000, 2447.5468750
4: -861.8063354, 1709.3990479, -723.2552490, 1447.1606445, -2308.9665527, 2432.6542969

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1030042, upper bound: 1757.1109154
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1030042, upper bound: 1757.1109153
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -342.4588623, 1389.2111816, -347.3078918, 1407.9704590, -1750.4290771, 1736.5190430
1: -419.8147888, 1550.8801270, -425.7395630, 1571.8404541, -1991.6552734, 1976.6196289
2: -480.9628296, 1576.8831787, -487.6861572, 1598.2337646, -2079.1965332, 2064.5693359
3: -682.6716309, 1720.9527588, -692.1408081, 1743.9346924, -2426.6062012, 2413.0935059
4: -808.7684937, 1604.4613037, -819.8779297, 1625.9879150, -2434.7563477, 2424.3393555

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1025262, upper bound: 1757.1025262
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1025262, upper bound: 1757.1025262
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -372.7439880, 1510.2344971, -348.6284180, 1412.4794922, -1785.2235107, 1858.8627930
1: -457.0692444, 1686.1851807, -427.2997437, 1576.9318848, -2034.0009766, 2113.4848633
2: -523.4106445, 1714.5540771, -489.4211121, 1603.4146729, -2126.8249512, 2203.9750977
3: -742.8453369, 1871.1206055, -694.5096436, 1749.2849121, -2492.1303711, 2565.6301270
4: -879.5175781, 1744.6900635, -822.4790649, 1631.1877441, -2510.7050781, 2567.1689453

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1025262, upper bound: 1757.1025262
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1025262, upper bound: 1757.1025262
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -296.5571289, 1203.9903564, -431.2108765, 1777.2193604, -2073.7763672, 1635.2009277
1: -363.4153442, 1344.1190186, -526.4514771, 1984.1525879, -2347.5671387, 1870.5705566
2: -415.7507935, 1366.2316895, -612.1891479, 2010.7648926, -2426.5151367, 1978.4206543
3: -590.0724487, 1488.4838867, -862.7782593, 2210.4677734, -2800.5402832, 2351.2622070
4: -697.3197021, 1389.4118652, -1038.7113037, 2047.3288574, -2744.6484375, 2428.1230469

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0761562, upper bound: 1757.0313660
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0842720, upper bound: 1757.0425772
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0842720, upper bound: 1757.0510307
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -308.5275269, 1251.8031006, -431.9667969, 1783.3555908, -2091.8830566, 1683.7698975
1: -378.0660706, 1397.6104736, -528.0718384, 1991.1435547, -2369.2087402, 1925.6821289
2: -432.4266357, 1420.6679688, -612.9890747, 2018.4073486, -2450.8339844, 2033.6569824
3: -613.9926758, 1548.1319580, -865.9387817, 2215.6435547, -2829.6362305, 2414.0708008
4: -725.7801514, 1444.6577148, -1040.4610596, 2053.7495117, -2779.5292969, 2485.1184082

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0696658, upper bound: 1757.0245818
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0696658, upper bound: 1757.0365321
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -334.6288452, 1358.0841064, -416.9645996, 1723.1430664, -2057.7717285, 1775.0485840
1: -410.2295837, 1516.1625977, -510.0505676, 1924.8194580, -2335.0485840, 2026.2131348
2: -470.1205139, 1541.5301514, -590.5448608, 1951.0938721, -2421.2143555, 2132.0744629
3: -667.1864624, 1682.5451660, -835.4343262, 2137.2204590, -2804.4069824, 2517.9787598
4: -790.7907715, 1568.3714600, -1000.1073608, 1983.9595947, -2774.7504883, 2568.4787598

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0816751, upper bound: 1757.0927992
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0843021, upper bound: 1757.0940959
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1031250, upper bound: 1757.0954890
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -365.0411987, 1479.7664795, -415.9324951, 1716.7507324, -2081.7919922, 1895.6988525
1: -447.6459656, 1652.1899414, -508.8409424, 1917.7159424, -2365.3613281, 2161.0307617
2: -512.7231445, 1679.9470215, -588.9649048, 1944.1726074, -2456.8957520, 2268.9118652
3: -727.6420898, 1833.4910889, -833.0627441, 2129.3039551, -2856.9460449, 2666.5537109
4: -861.8063354, 1709.3990479, -996.8743286, 1977.1065674, -2838.9128418, 2706.2734375

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0963763, upper bound: 1757.0925699
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1017727, upper bound: 1757.0956715
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -342.4588623, 1389.2111816, -442.9769592, 1827.0310059, -2169.4897461, 1832.1881104
1: -419.8147888, 1550.8801270, -541.8436890, 2039.9616699, -2459.7763672, 2092.7238770
2: -480.9628296, 1576.8831787, -628.2559814, 2068.1831055, -2549.1459961, 2205.1386719
3: -682.6716309, 1720.9527588, -888.0169678, 2269.2973633, -2951.9685059, 2608.9697266
4: -808.7684937, 1604.4613037, -1065.5311279, 2104.3444824, -2913.1130371, 2669.9924316

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0941204, upper bound: 1757.0533249
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1066581, upper bound: 1757.0946890
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -372.7439880, 1510.2344971, -441.7191772, 1819.6300049, -2192.3740234, 1951.9536133
1: -457.0692444, 1686.1851807, -540.3425903, 2031.7313232, -2488.8005371, 2226.5278320
2: -523.4106445, 1714.5540771, -626.3850098, 2060.1083984, -2583.5190430, 2340.9389648
3: -742.8453369, 1871.1206055, -885.1611938, 2260.1821289, -3003.0270996, 2756.2817383
4: -879.5175781, 1744.6900635, -1061.8116455, 2096.2919922, -2975.8095703, 2806.5017090

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0896415, upper bound: 1757.0530707
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1022828, upper bound: 1757.0946693
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -414.3720093, 1711.0965576, -342.0927734, 1386.7406006, -1801.1124268, 2053.1889648
1: -506.9606018, 1911.4222412, -419.3257751, 1548.2221680, -2055.1828613, 2330.7478027
2: -586.6476440, 1937.5845947, -480.4227295, 1574.1197510, -2160.7673340, 2418.0073242
3: -830.1010742, 2122.1179199, -681.6835938, 1717.6628418, -2547.7639160, 2803.8015137
4: -993.1072388, 1970.2883301, -807.7128296, 1601.3179932, -2594.4243164, 2778.0009766

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0927019, upper bound: 1757.0963868
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0927019, upper bound: 1757.0963868
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -419.0493469, 1729.8946533, -342.3240967, 1388.0311279, -1807.0804443, 2072.2180176
1: -512.6394653, 1932.3780518, -419.6032104, 1549.6734619, -2062.3129883, 2351.9812012
2: -593.2935791, 1959.0159912, -480.7357483, 1575.5661621, -2168.8593750, 2439.7517090
3: -839.3040161, 2145.3808594, -682.1703491, 1719.1496582, -2558.4536133, 2827.5512695
4: -1004.2643433, 1991.9449463, -808.2421875, 1602.7510986, -2607.0153809, 2800.1867676

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0949785, upper bound: 1757.1025662
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0956715, upper bound: 1757.1017727
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -414.3720093, 1711.0965576, -443.2364197, 1828.7536621, -2243.1252441, 2154.3327637
1: -506.9606018, 1911.4222412, -541.9617310, 2041.9305420, -2548.8908691, 2453.3840332
2: -586.6476440, 1937.5845947, -628.7308350, 2069.9167480, -2656.5644531, 2566.3154297
3: -830.1010742, 2122.1179199, -888.3505859, 2271.8015137, -3101.9025879, 3010.4685059
4: -993.1072388, 1970.2883301, -1066.5948486, 2106.2126465, -3099.3195801, 3036.8830566

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0926368, upper bound: 1757.0938145
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0926368, upper bound: 1757.0938145
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -419.0493469, 1729.8946533, -443.7115784, 1830.6767578, -2249.7253418, 2173.6057129
1: -512.6394653, 1932.3780518, -542.5518188, 2044.0916748, -2556.7309570, 2474.9299316
2: -593.2935791, 1959.0159912, -629.4076538, 2072.1047363, -2665.3984375, 2588.4235840
3: -839.3040161, 2145.3808594, -889.2767944, 2274.0729980, -3113.3769531, 3034.6569824
4: -1004.2643433, 1991.9449463, -1067.6551514, 2108.3903809, -3112.6547852, 3059.5996094

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0939388, upper bound: 1757.0864100
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0955349, upper bound: 1757.0944980
time: 0.77 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.44 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.1137332, upper bound: 1757.1170720
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.1137332, upper bound: 1757.1172965
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.1137332, upper bound: 1757.1170720
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.1137332, upper bound: 1757.1172965
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.0590093, upper bound: 1757.0919675
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.0590092, upper bound: 1757.0919675
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.1133992, upper bound: 1757.1107440
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.1133992, upper bound: 1757.1107440
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.1030042, upper bound: 1757.1109154
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.1030042, upper bound: 1757.1109153
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.1030042, upper bound: 1757.1109154
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.1030042, upper bound: 1757.1109153
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.1025262, upper bound: 1757.1025262
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.1025262, upper bound: 1757.1025262
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.1025262, upper bound: 1757.1025262
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.1025262, upper bound: 1757.1025262
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.0842720, upper bound: 1757.0425772
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.0842720, upper bound: 1757.0510307
IS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.0696658, upper bound: 1757.0245818
IS_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.0696658, upper bound: 1757.0365321
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.0843021, upper bound: 1757.0940959
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.1031250, upper bound: 1757.0954890
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.0963763, upper bound: 1757.0925699
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.1017727, upper bound: 1757.0956715
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.0941204, upper bound: 1757.0533249
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.1066581, upper bound: 1757.0946890
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.0896415, upper bound: 1757.0530707
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.1022828, upper bound: 1757.0946693
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.0927019, upper bound: 1757.0963868
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.0927019, upper bound: 1757.0963868
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.0949785, upper bound: 1757.1025662
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.0956715, upper bound: 1757.1017727
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.0926368, upper bound: 1757.0938145
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.0926368, upper bound: 1757.0938145
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.0939388, upper bound: 1757.0864100
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -1757.0955349, upper bound: 1757.0944980

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -272.6042480, 1111.4747314, -272.2157288, 1109.8139648, -1382.4182129, 1383.6901855
1: -334.2237854, 1241.4110107, -333.7474670, 1239.5623779, -1573.7860107, 1575.1584473
2: -381.0059814, 1261.2246094, -380.4518127, 1259.3488770, -1640.3546143, 1641.6762695
3: -542.2633057, 1371.0507812, -541.4631348, 1368.9881592, -1911.2514648, 1912.5139160
4: -637.6555786, 1281.7025146, -636.7003784, 1279.8031006, -1917.4584961, 1918.4028320

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1008577, upper bound: 1757.1139883
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1175045, upper bound: 1757.1173555
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -272.6042480, 1111.4747314, -308.8294983, 1256.3496094, -1528.9538574, 1420.3040771
1: -334.2237854, 1241.4110107, -378.8326416, 1403.4211426, -1737.6448975, 1620.2435303
2: -381.0059814, 1261.2246094, -432.2922058, 1426.1948242, -1807.2008057, 1693.5168457
3: -542.2633057, 1371.0507812, -615.2606201, 1552.2734375, -2094.5366211, 1986.3112793
4: -637.6555786, 1281.7025146, -724.8505859, 1449.9963379, -2087.6518555, 2006.5529785

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1008577, upper bound: 1757.1139883
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1175045, upper bound: 1757.1176677
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -311.7570190, 1265.6446533, -272.2157288, 1109.8139648, -1421.5708008, 1537.8603516
1: -381.9933777, 1413.0380859, -333.7474670, 1239.5623779, -1621.5552979, 1746.7855225
2: -436.8719788, 1436.2128906, -380.4518127, 1259.3488770, -1696.2205811, 1816.6645508
3: -620.3704224, 1564.9118652, -541.4631348, 1368.9881592, -1989.3586426, 2106.3750000
4: -733.2883911, 1460.4486084, -636.7003784, 1279.8031006, -2013.0915527, 2097.1489258

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1047023, upper bound: 1757.1147351
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1137332, upper bound: 1757.1170720
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -311.7570190, 1265.6446533, -308.8294983, 1256.3496094, -1568.1065674, 1574.4741211
1: -381.9933777, 1413.0380859, -378.8326416, 1403.4211426, -1785.4143066, 1791.8706055
2: -436.8719788, 1436.2128906, -432.2922058, 1426.1948242, -1863.0666504, 1868.5051270
3: -620.3704224, 1564.9118652, -615.2606201, 1552.2734375, -2172.6437988, 2180.1721191
4: -733.2883911, 1460.4486084, -724.8505859, 1449.9963379, -2183.2839355, 2185.2988281

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1047023, upper bound: 1757.1147371
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1137332, upper bound: 1757.1172965
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -306.9274597, 1237.4173584, -303.2347412, 1231.4592285, -1538.3867188, 1540.6519775
1: -375.7303467, 1382.0565186, -371.6198425, 1374.9683838, -1750.6987305, 1753.6761475
2: -429.6966553, 1404.1936035, -424.6167908, 1397.4414062, -1827.1380615, 1828.8101807
3: -609.3472900, 1534.1628418, -603.2944336, 1521.9353027, -2131.2824707, 2137.4572754
4: -721.7412109, 1429.1094971, -712.1339111, 1421.0101318, -2142.7514648, 2141.2431641

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0431014, upper bound: 1757.0774896
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0573635, upper bound: 1757.0919675
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0590092, upper bound: 1757.0916188
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -306.9274597, 1237.4173584, -340.4002686, 1380.1796875, -1687.1071777, 1577.8176270
1: -375.7303467, 1382.0565186, -417.3317261, 1540.9694824, -1916.6997070, 1799.3881836
2: -429.6966553, 1404.1936035, -477.6791077, 1566.7270508, -1996.4237061, 1881.8726807
3: -609.3472900, 1534.1628418, -678.2658081, 1708.4897461, -2317.8369141, 2212.4287109
4: -721.7412109, 1429.1094971, -802.4424438, 1593.8431396, -2315.5844727, 2231.5515137

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0431014, upper bound: 1757.0774922
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0573635, upper bound: 1757.0919675
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0590092, upper bound: 1757.0916188
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -308.8428345, 1253.4255371, -309.8702698, 1257.7979736, -1566.6407471, 1563.2957764
1: -378.4500122, 1399.4350586, -379.6989746, 1404.2880859, -1782.7380371, 1779.1340332
2: -432.7960205, 1422.3830566, -434.2360229, 1427.3289795, -1860.1250000, 1856.6190186
3: -614.5263062, 1549.8706055, -616.6035767, 1555.2449951, -2169.7712402, 2166.4741211
4: -726.3590698, 1446.4643555, -728.8213501, 1451.4553223, -2177.8144531, 2175.2854004

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0641076, upper bound: 1757.0942665
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1055134, upper bound: 1757.1017912
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -308.8428345, 1253.4255371, -347.7091370, 1408.8258057, -1717.6687012, 1601.1346436
1: -378.4500122, 1399.4350586, -426.2081299, 1572.8552246, -1951.3051758, 1825.6431885
2: -432.7960205, 1422.3830566, -488.1999817, 1599.2493896, -2032.0452881, 1910.5827637
3: -614.5263062, 1549.8706055, -692.7598877, 1744.8815918, -2359.4079590, 2242.6303711
4: -726.3590698, 1446.4643555, -820.5458984, 1626.9672852, -2353.3259277, 2267.0102539

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0641076, upper bound: 1757.0942665
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1055134, upper bound: 1757.1017912
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -334.6288452, 1358.0841064, -302.6579285, 1232.5910645, -1567.2199707, 1660.7420654
1: -410.2295837, 1516.1625977, -371.3000183, 1376.8480225, -1787.0773926, 1887.4626465
2: -470.1205139, 1541.5301514, -423.7609558, 1399.1936035, -1869.3140869, 1965.2911377
3: -667.1864624, 1682.5451660, -603.3029175, 1523.3959961, -2190.5825195, 2285.8476562
4: -790.7907715, 1568.3714600, -710.8558350, 1423.0701904, -2213.8608398, 2279.2270508

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1091949, upper bound: 1757.1113645
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1091949, upper bound: 1757.1113645
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -334.6288452, 1358.0841064, -331.8717651, 1349.4671631, -1684.0959473, 1689.9558105
1: -410.2295837, 1516.1625977, -407.2701416, 1507.4794922, -1917.7089844, 1923.4326172
2: -470.1205139, 1541.5301514, -464.8316040, 1532.1937256, -2002.3140869, 2006.3616943
3: -667.1864624, 1682.5451660, -661.4526367, 1668.5430908, -2335.7294922, 2343.9970703
4: -790.7907715, 1568.3714600, -779.4064941, 1558.6022949, -2349.3928223, 2347.7778320

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1091949, upper bound: 1757.1113645
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1091949, upper bound: 1757.1113645
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -365.0411987, 1479.7664795, -302.6579285, 1232.5910645, -1597.6317139, 1782.4244385
1: -447.6459656, 1652.1899414, -371.3000183, 1376.8480225, -1824.4940186, 2023.4899902
2: -512.7231445, 1679.9470215, -423.7609558, 1399.1936035, -1911.9167480, 2103.7080078
3: -727.6420898, 1833.4910889, -603.3029175, 1523.3959961, -2251.0380859, 2436.7939453
4: -861.8063354, 1709.3990479, -710.8558350, 1423.0701904, -2284.8759766, 2420.2548828

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1030042, upper bound: 1757.1109154
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1030042, upper bound: 1757.1109153
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -365.0411987, 1479.7664795, -331.8717651, 1349.4671631, -1714.5078125, 1811.6381836
1: -447.6459656, 1652.1899414, -407.2701416, 1507.4794922, -1955.1254883, 2059.4599609
2: -512.7231445, 1679.9470215, -464.8316040, 1532.1937256, -2044.9168701, 2144.7785645
3: -727.6420898, 1833.4910889, -661.4526367, 1668.5430908, -2396.1850586, 2494.9436035
4: -861.8063354, 1709.3990479, -779.4064941, 1558.6022949, -2420.4082031, 2488.8056641

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1030042, upper bound: 1757.1109154
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1030042, upper bound: 1757.1109154
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -342.4588623, 1389.2111816, -342.8592834, 1390.5733643, -1733.0321045, 1732.0704346
1: -419.8147888, 1550.8801270, -420.3100891, 1552.3649902, -1972.1798096, 1971.1901855
2: -480.9628296, 1576.8831787, -481.5136414, 1578.4736328, -2059.4362793, 2058.3962402
3: -682.6716309, 1720.9527588, -683.4462280, 1722.6535645, -2405.3251953, 2404.3989258
4: -808.7684937, 1604.4613037, -809.6674805, 1606.0499268, -2414.8183594, 2414.1289062

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0549486, upper bound: 1757.0847965
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1083502, upper bound: 1757.1027868
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -342.4588623, 1389.2111816, -373.1291199, 1511.5972900, -1854.0561523, 1762.3402100
1: -419.8147888, 1550.8801270, -457.5401611, 1687.6634521, -2107.4782715, 2008.4202881
2: -480.9628296, 1576.8831787, -523.9413452, 1716.1180420, -2197.0805664, 2100.8244629
3: -682.6716309, 1720.9527588, -743.6129761, 1872.8608398, -2555.5322266, 2464.5656738
4: -808.7684937, 1604.4613037, -880.4192505, 1746.2985840, -2555.0668945, 2484.8806152

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0549486, upper bound: 1757.0847965
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1083502, upper bound: 1757.1027868
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -372.7439880, 1510.2344971, -342.8592834, 1390.5733643, -1763.3173828, 1853.0937500
1: -457.0692444, 1686.1851807, -420.3100891, 1552.3649902, -2009.4342041, 2106.4953613
2: -523.4106445, 1714.5540771, -481.5136414, 1578.4736328, -2101.8840332, 2196.0676270
3: -742.8453369, 1871.1206055, -683.4462280, 1722.6535645, -2465.4990234, 2554.5668945
4: -879.5175781, 1744.6900635, -809.6674805, 1606.0499268, -2485.5673828, 2554.3574219

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1025262, upper bound: 1757.1025262
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1025262, upper bound: 1757.1025262
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -372.7439880, 1510.2344971, -373.1291199, 1511.5972900, -1884.3413086, 1883.3635254
1: -457.0692444, 1686.1851807, -457.5401611, 1687.6634521, -2144.7326660, 2143.7253418
2: -523.4106445, 1714.5540771, -523.9413452, 1716.1180420, -2239.5280762, 2238.4953613
3: -742.8453369, 1871.1206055, -743.6129761, 1872.8608398, -2615.7058105, 2614.7336426
4: -879.5175781, 1744.6900635, -880.4192505, 1746.2985840, -2625.8159180, 2625.1093750

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1025262, upper bound: 1757.1025262
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1025262, upper bound: 1757.1025262
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -296.5571289, 1203.9903564, -420.7312012, 1732.6887207, -2029.2458496, 1624.7215576
1: -363.4153442, 1344.1190186, -513.0576782, 1934.4208984, -2297.8356934, 1857.1767578
2: -415.7507935, 1366.2316895, -597.2897339, 1960.4376221, -2376.1882324, 1963.5214844
3: -590.0724487, 1488.4838867, -840.9810181, 2156.6936035, -2746.7661133, 2329.4648438
4: -697.3197021, 1389.4118652, -1013.7831421, 1996.4489746, -2693.7685547, 2403.1945801

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0842720, upper bound: 1757.0425772
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0818917, upper bound: 1757.0423333
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -296.5571289, 1203.9903564, -430.1464844, 1772.6765137, -2069.2336426, 1634.1367188
1: -363.4153442, 1344.1190186, -525.1240845, 1979.0767822, -2342.4916992, 1869.2431641
2: -415.7507935, 1366.2316895, -610.6856689, 2005.6040039, -2421.3542480, 1976.9173584
3: -590.0724487, 1488.4838867, -860.6032715, 2204.9443359, -2795.0168457, 2349.0869141
4: -697.3197021, 1389.4118652, -1036.2095947, 2042.0972900, -2739.4169922, 2425.6213379

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0842720, upper bound: 1757.0510307
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0818917, upper bound: 1757.0508946
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -327.3001709, 1327.1610107, -411.0552979, 1698.7313232, -2026.0311279, 1738.2163086
1: -401.2607727, 1481.6027832, -502.9038696, 1897.1894531, -2298.4501953, 1984.5065918
2: -459.8415222, 1506.4490967, -581.8116455, 1923.7966309, -2383.6381836, 2088.2604980
3: -652.5797119, 1644.6595459, -823.6427612, 2105.8835449, -2758.4628906, 2468.3022461
4: -773.6840820, 1532.7086182, -985.2358398, 1955.6722412, -2729.3557129, 2517.9443359

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 29

Time for candidate selection: 7.92 seconds

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0790481, upper bound: 1757.0931942
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0842856, upper bound: 1757.0931373
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -332.3886719, 1348.8437500, -410.2469788, 1695.4487305, -2027.8374023, 1759.0906982
1: -407.5131226, 1505.8470459, -501.9011841, 1893.8803711, -2301.3930664, 2007.7482910
2: -466.9366455, 1531.0805664, -581.0557861, 1919.7940674, -2386.7307129, 2112.1362305
3: -662.7254028, 1671.0706787, -822.0683594, 2102.7976074, -2765.5229492, 2493.1389160
4: -785.3336792, 1557.8044434, -983.9697876, 1952.1129150, -2737.4455566, 2541.7736816

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1013181, upper bound: 1757.0946873
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1025662, upper bound: 1757.0949785
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -363.7440796, 1474.2481689, -409.2890625, 1689.6806641, -2053.4248047, 1883.5371094
1: -446.0682068, 1646.0114746, -500.7733765, 1887.5070801, -2333.5751953, 2146.7841797
2: -510.8954773, 1673.6984863, -579.5447388, 1913.3980713, -2424.2934570, 2253.2429199
3: -725.0507202, 1826.7346191, -819.9064331, 2095.7375488, -2820.7880859, 2646.6411133
4: -858.7074585, 1703.0935059, -980.9437866, 1945.9482422, -2804.6557617, 2684.0373535

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0963763, upper bound: 1757.0925699
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0963763, upper bound: 1757.0925699
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -363.9776917, 1475.4960938, -413.9446411, 1708.4090576, -2072.3864746, 1889.4405518
1: -446.3529358, 1647.4194336, -506.4268799, 1908.3840332, -2354.7370605, 2153.8461914
2: -511.2203674, 1675.0976562, -586.1633911, 1934.7562256, -2445.9765625, 2261.2609863
3: -725.5415649, 1828.1723633, -829.0708008, 2118.9082031, -2844.4497070, 2657.2431641
4: -859.2610474, 1704.4832764, -992.0501099, 1967.5266113, -2826.7863770, 2696.5327148

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1017727, upper bound: 1757.0956715
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1017727, upper bound: 1757.0956715
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -333.1293335, 1352.0805664, -433.2927856, 1783.9381104, -2117.0673828, 1785.3732910
1: -408.4671326, 1509.5345459, -529.4693604, 1991.8175049, -2400.2846680, 2039.0039062
2: -467.6218872, 1534.7746582, -614.4465332, 2019.7421875, -2487.3640137, 2149.2204590
3: -664.1210938, 1674.1690674, -867.8065796, 2218.1127930, -2882.2338867, 2541.9753418
4: -785.8925781, 1561.6707764, -1042.5911865, 2055.9794922, -2841.8720703, 2604.2619629

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0535127, upper bound: 1757.0511304
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0535127, upper bound: 1757.0533249
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -340.3804321, 1380.5463867, -440.5360107, 1817.0747070, -2157.4550781, 1821.0823975
1: -417.2844543, 1541.2158203, -538.8500366, 2028.8645020, -2446.1489258, 2080.0659180
2: -478.0729065, 1567.0839844, -624.8539429, 2056.8254395, -2534.8984375, 2191.9379883
3: -678.5205078, 1710.2846680, -883.1613159, 2257.1394043, -2935.6599121, 2593.4460449
4: -803.8822021, 1594.5225830, -1059.9062500, 2092.8720703, -2896.7543945, 2654.4282227

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1031250, upper bound: 1757.0946890
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1031250, upper bound: 1757.0946890
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -363.3578491, 1472.6871338, -431.7387695, 1775.4205322, -2138.7780762, 1904.4259033
1: -445.6650085, 1644.3631592, -527.6112061, 1982.3281250, -2427.9931641, 2171.9743652
2: -510.0274353, 1672.0136719, -612.1722412, 2010.3150635, -2520.3425293, 2284.1860352
3: -724.1702271, 1823.8640137, -864.3948364, 2207.6306152, -2931.8007812, 2688.2585449
4: -856.5534058, 1701.4345703, -1038.2161865, 2046.5904541, -2903.1437988, 2739.6508789

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0896415, upper bound: 1757.0530709
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 38

Time for candidate selection: 8.58 seconds

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0797122, upper bound: 1757.0395150
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0572923, upper bound: 1757.0373190
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -370.6557007, 1501.5709229, -439.1918945, 1809.2873535, -2179.9431152, 1940.7624512
1: -454.5268555, 1676.5258789, -537.2446289, 2020.1983643, -2474.7250977, 2213.7700195
2: -520.5037842, 1704.7528076, -622.8616333, 2048.2978516, -2568.8017578, 2327.6145020
3: -738.6829224, 1860.4573975, -880.1318970, 2247.5529785, -2986.2353516, 2740.5888672
4: -874.6104126, 1734.7634277, -1055.9873047, 2084.3759766, -2958.9863281, 2790.7504883

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1022828, upper bound: 1757.0946693
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1022828, upper bound: 1757.0946693
time: 0.74 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.48 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1008577, upper bound: 1757.1139883
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1175045, upper bound: 1757.1173555
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1008577, upper bound: 1757.1139883
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1175045, upper bound: 1757.1176677
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1047023, upper bound: 1757.1147351
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1137332, upper bound: 1757.1170720
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1047023, upper bound: 1757.1147371
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1137332, upper bound: 1757.1172965
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.0573635, upper bound: 1757.0919675
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.0590092, upper bound: 1757.0916188
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.0573635, upper bound: 1757.0919675
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.0590092, upper bound: 1757.0916188
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.0641076, upper bound: 1757.0942665
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1055134, upper bound: 1757.1017912
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.0641076, upper bound: 1757.0942665
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1055134, upper bound: 1757.1017912
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1091949, upper bound: 1757.1113645
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1091949, upper bound: 1757.1113645
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1091949, upper bound: 1757.1113645
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1091949, upper bound: 1757.1113645
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1030042, upper bound: 1757.1109154
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1030042, upper bound: 1757.1109153
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1030042, upper bound: 1757.1109154
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1030042, upper bound: 1757.1109154
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.0549486, upper bound: 1757.0847965
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1083502, upper bound: 1757.1027868
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.0549486, upper bound: 1757.0847965
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1083502, upper bound: 1757.1027868
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1025262, upper bound: 1757.1025262
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1025262, upper bound: 1757.1025262
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1025262, upper bound: 1757.1025262
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1025262, upper bound: 1757.1025262
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.0842720, upper bound: 1757.0425772
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.0818917, upper bound: 1757.0423333
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.0842720, upper bound: 1757.0510307
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.0818917, upper bound: 1757.0508946
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.0790481, upper bound: 1757.0931942
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.0842856, upper bound: 1757.0931373
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1013181, upper bound: 1757.0946873
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1025662, upper bound: 1757.0949785
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.0963763, upper bound: 1757.0925699
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.0963763, upper bound: 1757.0925699
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1017727, upper bound: 1757.0956715
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1017727, upper bound: 1757.0956715
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.0535127, upper bound: 1757.0511304
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.0535127, upper bound: 1757.0533249
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1031250, upper bound: 1757.0946890
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1031250, upper bound: 1757.0946890
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.0797122, upper bound: 1757.0395150
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.0572923, upper bound: 1757.0373190
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1022828, upper bound: 1757.0946693
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.48
Output dim: 0, lower bound: -1757.1022828, upper bound: 1757.0946693
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -1757.0927019, upper bound: 1757.0963868
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -1757.0927019, upper bound: 1757.0963868
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -1757.0949785, upper bound: 1757.1025662
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -1757.0956715, upper bound: 1757.1017727
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -1757.0926368, upper bound: 1757.0938145
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -1757.0926368, upper bound: 1757.0938145
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -1757.0939388, upper bound: 1757.0864100
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -1757.0955349, upper bound: 1757.0944980
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=2002.608154296875
rel_dist={0: [-1757.1254498169274, 1757.1254498169274]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1132.57 seconds
