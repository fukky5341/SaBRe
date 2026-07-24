## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_1.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 1781.702970027904


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-410.8436279, 1662.2098389, -410.8436279, 1662.2098389, -2073.0534668, 2073.0534668)
1: (-661.0173340, 1839.7109375, -661.0173340, 1839.7109375, -2500.7282715, 2500.7282715)
2: (-493.7767944, 2118.4650879, -493.7767944, 2118.4650879, -2612.2416992, 2612.2416992)
3: (-1061.6599121, 1896.6071777, -1061.6599121, 1896.6071777, -2958.2668457, 2958.2668457)
4: (-851.2732544, 1983.1082764, -851.2732544, 1983.1082764, -2834.3815918, 2834.3815918)

## BASE Result
execution time: IAR + LP analysis = 1.30 + 1.72 = 3.02 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1781.7419244, upper bound: 1781.7419244


# Binary Search by BASE starts (time budget: 1196.98 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=2073.053466796875
rel_dist={0: [-1781.7419244201583, 1781.7419244201583]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=2073.053466796875
rel_dist={0: [-1781.7403846768325, 1781.740384676833]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=2073.053466796875
rel_dist={0: [-1781.7388006996591, 1781.738800699659]}

## Binary search (step 3) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=2073.053466796875
rel_dist={0: [-1781.7379273494657, 1781.7379273494657]}

## Binary search (step 4) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=2073.053466796875
rel_dist={0: [-1781.7373817342982, 1781.7373817342977]}

## Binary search (step 5) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=2073.053466796875
rel_dist={0: [-1781.737074999111, 1781.7370749991114]}

## Binary search (step 6) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=2073.053466796875
rel_dist={0: [-1781.7369176631933, 1781.7369176631928]}

## Binary search (step 7) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=2073.053466796875
rel_dist={0: [-1781.7368389945823, 1781.7368389945823]}

## Binary search (step 8) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=2073.053466796875
rel_dist={0: [-1781.7367996602775, 1781.736799660277]}

## Binary search (step 9) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=2073.053466796875
rel_dist={0: [-1781.7367799905142, 1781.7367799905142]}

## Binary search (step 10) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=2073.053466796875
rel_dist={0: [-1781.736770119484, 1781.7367701194835]}

## Binary search (step 11) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=2073.053466796875
rel_dist={0: [-1781.7367651839704, 1781.7367651839704]}

## Binary search (step 12) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=2073.053466796875
rel_dist={0: [-1781.7367627162182, 1781.7367627162175]}

## Binary search (step 13) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=2073.053466796875
rel_dist={0: [-1781.7367614823506, 1781.7367614823506]}

## Binary search (step 14) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=2073.053466796875
rel_dist={0: [-1781.7367608654336, 1781.736760865433]}

## Binary search (step 15) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=2073.053466796875
rel_dist={0: [-1781.7367605577476, 1781.736760557009]}

## Binary search (step 16) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=2073.053466796875
rel_dist={0: [-1781.736760403091, 1781.7367604030915]}

## Binary search (step 17) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=2073.053466796875
rel_dist={0: [-1781.7367603258897, 1781.73676032589]}

## Binary search (step 18) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=2073.053466796875
rel_dist={0: [-1781.736760288989, 1781.73676029242]}

## Binary search (step 19) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 19): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=2073.053466796875
rel_dist={0: [-1781.7367602804184, 1781.7367602745474]}

## Binary Search Result
Binary search time: 61.61 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1135.36 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7419244, upper bound: 1781.7419244
time: 0.54 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7419244, upper bound: 1781.7419244
time: 1.93 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.60 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.60
Output dim: 0, lower bound: -1781.7419244, upper bound: 1781.7419244
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.60
Output dim: 0, lower bound: -1781.7419244, upper bound: 1781.7419244

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -292.2078247, 1185.7346191, -409.8881226, 1658.3120117, -1950.5197754, 1595.6226807
1: -470.8653564, 1312.1569824, -659.4754639, 1835.4072266, -2306.2724609, 1971.6322021
2: -351.4121094, 1511.5142822, -492.6187134, 2113.4978027, -2464.9099121, 2004.1329346
3: -755.9409180, 1351.1541748, -1059.2000732, 1892.1563721, -2648.0971680, 2410.3542480
4: -605.2770386, 1414.5281982, -849.2722778, 1978.4670410, -2583.7441406, 2263.8005371

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7404735, upper bound: 1781.7403250
time: 0.53 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7405927, upper bound: 1781.7405927
time: 0.63 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -407.7161560, 1648.8684082, -410.8436279, 1662.2098389, -2069.9260254, 2059.7119141
1: -656.0112915, 1825.3117676, -661.0173340, 1839.7109375, -2495.7221680, 2486.3291016
2: -490.0357056, 2101.4226074, -493.7767944, 2118.4650879, -2608.5007324, 2595.1992188
3: -1053.7666016, 1881.9282227, -1061.6599121, 1896.6071777, -2950.3735352, 2943.5881348
4: -844.8474731, 1967.4259033, -851.2732544, 1983.1082764, -2827.9558105, 2818.6989746

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7419244, upper bound: 1781.7419244
time: 0.52 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7419244, upper bound: 1781.7419244
time: 0.58 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.36 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 0, lower bound: -1781.7404735, upper bound: 1781.7403250
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 0, lower bound: -1781.7405927, upper bound: 1781.7405927
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 0, lower bound: -1781.7419244, upper bound: 1781.7419244
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 0, lower bound: -1781.7419244, upper bound: 1781.7419244

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -292.2078247, 1185.7346191, -381.5317993, 1543.0340576, -1835.2418213, 1567.2663574
1: -470.8653564, 1312.1569824, -614.1837158, 1708.1824951, -2179.0478516, 1926.3406982
2: -351.4121094, 1511.5142822, -458.8406982, 1966.9923096, -2318.4042969, 1970.3549805
3: -755.9409180, 1351.1541748, -986.2930908, 1761.2056885, -2517.1464844, 2337.4472656
4: -605.2770386, 1414.5281982, -791.3146362, 1841.0732422, -2446.3503418, 2205.8427734

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7160553, upper bound: 1781.7290461
time: 0.54 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7110502, upper bound: 1781.7192287
time: 0.55 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -292.0651245, 1185.1374512, -436.4287109, 1765.1412354, -2057.2062988, 1621.5661621
1: -470.6372681, 1311.5079346, -701.5631104, 1954.6235352, -2425.2600098, 2013.0709229
2: -351.2402954, 1510.7570801, -524.1991577, 2250.1591797, -2601.3994141, 2034.9561768
3: -755.5812988, 1350.4630127, -1128.4348145, 2012.9937744, -2768.5747070, 2478.8972168
4: -604.9828491, 1413.8060303, -904.2096558, 2104.9616699, -2709.9445801, 2318.0156250

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7161104, upper bound: 1781.7294011
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7070154, upper bound: 1781.7070662
time: 0.55 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -407.7161560, 1648.8684082, -292.2078247, 1185.7346191, -1593.4508057, 1941.0761719
1: -656.0112915, 1825.3117676, -470.8653564, 1312.1569824, -1968.1682129, 2296.1772461
2: -490.0357056, 2101.4226074, -351.4121094, 1511.5142822, -2001.5499268, 2452.8347168
3: -1053.7666016, 1881.9282227, -755.9409180, 1351.1541748, -2404.9206543, 2637.8691406
4: -844.8474731, 1967.4259033, -605.2770386, 1414.5281982, -2259.3757324, 2572.7026367

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7403250, upper bound: 1781.7404735
time: 0.48 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7405927, upper bound: 1781.7405927
time: 0.53 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -407.7161560, 1648.8684082, -407.7161560, 1648.8684082, -2056.5844727, 2056.5844727
1: -656.0112915, 1825.3117676, -656.0112915, 1825.3117676, -2481.3229980, 2481.3229980
2: -490.0357056, 2101.4226074, -490.0357056, 2101.4226074, -2591.4582520, 2591.4582520
3: -1053.7666016, 1881.9282227, -1053.7666016, 1881.9282227, -2935.6948242, 2935.6948242
4: -844.8474731, 1967.4259033, -844.8474731, 1967.4259033, -2812.2729492, 2812.2729492

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7403250, upper bound: 1781.7404735
time: 0.49 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7405927, upper bound: 1781.7405927
time: 0.51 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.32 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -1781.7160553, upper bound: 1781.7290461
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -1781.7110502, upper bound: 1781.7192287
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -1781.7161104, upper bound: 1781.7294011
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -1781.7070154, upper bound: 1781.7070662
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -1781.7403250, upper bound: 1781.7404735
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -1781.7405927, upper bound: 1781.7405927
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -1781.7403250, upper bound: 1781.7404735
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -1781.7405927, upper bound: 1781.7405927

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -292.2078247, 1185.7346191, -367.4099121, 1486.1654053, -1778.3731689, 1553.1442871
1: -470.8653564, 1312.1569824, -591.3910522, 1645.4392090, -2116.3044434, 1903.5480957
2: -351.4121094, 1511.5142822, -441.6774902, 1894.3797607, -2245.7917480, 1953.1917725
3: -755.9409180, 1351.1541748, -949.8151855, 1695.6885986, -2451.6293945, 2300.9692383
4: -605.2770386, 1414.5281982, -761.6368408, 1772.8020020, -2378.0791016, 2176.1650391

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7151232, upper bound: 1781.7287093
time: 0.54 seconds

## Relational analysis of IS_A1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7110502, upper bound: 1781.7192287
time: 0.59 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7110502, upper bound: 1781.7192287
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -289.0429993, 1172.7789307, -463.9091797, 1858.9315186, -2147.9736328, 1636.6881104
1: -465.7215576, 1298.0424805, -749.8897705, 2063.9079590, -2529.6293945, 2047.9320068
2: -347.5960388, 1495.0695801, -560.7911377, 2372.5187988, -2720.1147461, 2055.8608398
3: -747.8043823, 1336.4666748, -1207.1788330, 2130.0898438, -2877.8935547, 2543.6452637
4: -598.7250366, 1399.0037842, -965.7495117, 2224.1687012, -2822.8935547, 2364.7534180

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7107829, upper bound: 1781.7178425
time: 0.69 seconds

## Relational analysis of IS_A1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7110502, upper bound: 1781.7180200
time: 0.47 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -292.0651245, 1185.1374512, -421.4374084, 1704.4232178, -1996.4882812, 1606.5748291
1: -470.6372681, 1311.5079346, -677.4370117, 1887.7091064, -2358.3459473, 1988.9449463
2: -351.2402954, 1510.7570801, -506.1168823, 2172.7185059, -2523.9587402, 2016.8740234
3: -755.5812988, 1350.4630127, -1089.7391357, 1943.7270508, -2699.3081055, 2440.2019043
4: -604.9828491, 1413.8060303, -873.0332642, 2032.4429932, -2637.4257812, 2286.8391113

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7161104, upper bound: 1781.7294011
time: 0.82 seconds

## Relational analysis of IS_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7161104, upper bound: 1781.7294011
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -288.8876648, 1172.1311035, -487.8399963, 1958.5509033, -2247.4384766, 1659.9710693
1: -465.4729919, 1297.3371582, -786.8693848, 2173.6418457, -2639.1147461, 2084.2062988
2: -347.4084778, 1494.2482910, -588.7560425, 2498.6284180, -2846.0368652, 2083.0041504
3: -747.4121094, 1335.7172852, -1267.0052490, 2243.1367188, -2990.5483398, 2602.7226562
4: -598.4033203, 1398.2211914, -1014.6013794, 2341.4963379, -2939.8996582, 2412.8215332

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7070154, upper bound: 1781.7070662
time: 0.70 seconds

## Relational analysis of IS_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7070154, upper bound: 1781.7070662
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -380.0692139, 1536.6132812, -292.2078247, 1185.7346191, -1565.8037109, 1828.8210449
1: -611.8825684, 1701.3682861, -470.8653564, 1312.1569824, -1924.0395508, 2172.2331543
2: -457.1301575, 1958.7752686, -351.4121094, 1511.5142822, -1968.6442871, 2310.1875000
3: -982.6779175, 1754.2751465, -755.9409180, 1351.1541748, -2333.8315430, 2510.2158203
4: -788.3563232, 1833.5460205, -605.2770386, 1414.5281982, -2202.8842773, 2438.8229980

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7290461, upper bound: 1781.7160553
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7192287, upper bound: 1781.7110502
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -433.4168091, 1752.8944092, -292.0651245, 1185.1374512, -1618.5541992, 2044.9594727
1: -696.7171021, 1941.2100830, -470.6372681, 1311.5079346, -2008.2249756, 2411.8474121
2: -520.5569458, 2234.4389648, -351.2402954, 1510.7570801, -2031.3139648, 2585.6791992
3: -1120.6684570, 1999.0828857, -755.5812988, 1350.4630127, -2471.1313477, 2754.6640625
4: -897.9285889, 2090.1264648, -604.9828491, 1413.8060303, -2311.7346191, 2695.1093750

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7294011, upper bound: 1781.7161104
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7070662, upper bound: 1781.7070154
time: 0.48 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -380.0692139, 1536.6132812, -407.7161560, 1648.8684082, -2028.9375000, 1944.3294678
1: -611.8825684, 1701.3682861, -656.0112915, 1825.3117676, -2437.1943359, 2357.3791504
2: -457.1301575, 1958.7752686, -490.0357056, 2101.4226074, -2558.5527344, 2448.8110352
3: -982.6779175, 1754.2751465, -1053.7666016, 1881.9282227, -2864.6062012, 2808.0412598
4: -788.3563232, 1833.5460205, -844.8474731, 1967.4259033, -2755.7817383, 2678.3935547

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7401786, upper bound: 1781.7402057
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7401786, upper bound: 1781.7404735
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -433.4168091, 1752.8944092, -407.6167908, 1648.4715576, -2081.8884277, 2160.5112305
1: -696.7171021, 1941.2100830, -655.8494263, 1824.8702393, -2521.5874023, 2597.0595703
2: -520.5569458, 2234.4389648, -489.9127502, 2100.9128418, -2621.4694824, 2724.3515625
3: -1120.6684570, 1999.0828857, -1053.5073242, 1881.4702148, -3002.1386719, 3052.5903320
4: -897.9285889, 2090.1264648, -844.6366577, 1966.9448242, -2864.8728027, 2934.7631836

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7294011, upper bound: 1781.7161104
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7070154, upper bound: 1781.7070154
time: 0.53 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.47 seconds
IS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -1781.7110502, upper bound: 1781.7192287
IS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -1781.7110502, upper bound: 1781.7192287
IS_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -1781.7107829, upper bound: 1781.7178425
IS_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -1781.7110502, upper bound: 1781.7180200
IS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -1781.7161104, upper bound: 1781.7294011
IS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -1781.7161104, upper bound: 1781.7294011
IS_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -1781.7070154, upper bound: 1781.7070662
IS_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -1781.7070154, upper bound: 1781.7070662
IS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -1781.7290461, upper bound: 1781.7160553
IS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -1781.7192287, upper bound: 1781.7110502
IS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -1781.7294011, upper bound: 1781.7161104
IS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -1781.7070662, upper bound: 1781.7070154
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -1781.7401786, upper bound: 1781.7402057
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -1781.7401786, upper bound: 1781.7404735
IS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -1781.7294011, upper bound: 1781.7161104
IS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -1781.7070154, upper bound: 1781.7070154

## BFS IS instance: IS_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -276.7585144, 1124.4237061, -367.4099121, 1486.1654053, -1762.9238281, 1491.8334961
1: -446.0162659, 1244.2011719, -591.3910522, 1645.4392090, -2091.4548340, 1835.5922852
2: -332.7811279, 1433.4194336, -441.6774902, 1894.3797607, -2227.1604004, 1875.0969238
3: -715.9465942, 1280.3992920, -949.8151855, 1695.6885986, -2411.6352539, 2230.2143555
4: -573.1133423, 1340.8247070, -761.6368408, 1772.8020020, -2345.9152832, 2102.4614258

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7151232, upper bound: 1781.7287093
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7159641, upper bound: 1781.7269907
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7160553, upper bound: 1781.7266019
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -371.8542480, 1491.4521484, -367.4099121, 1486.1654053, -1858.0196533, 1858.8618164
1: -601.9755249, 1656.0629883, -591.3910522, 1645.4392090, -2247.4147949, 2247.4541016
2: -449.9300232, 1903.8698730, -441.6774902, 1894.3797607, -2344.3098145, 2345.5473633
3: -969.7329712, 1708.7253418, -949.8151855, 1695.6885986, -2665.4216309, 2658.5405273
4: -774.2708130, 1785.2263184, -761.6368408, 1772.8020020, -2547.0727539, 2546.8630371

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7151232, upper bound: 1781.7287093
time: 0.54 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B1_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7159641, upper bound: 1781.7269907
time: 0.48 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2

### Relational analysis result of IS_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7160553, upper bound: 1781.7266019
time: 0.48 seconds

## BFS IS instance: IS_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -289.0429993, 1172.7789307, -454.6191711, 1819.8804932, -2108.9235840, 1627.3979492
1: -465.7215576, 1298.0424805, -735.1511841, 2021.0734863, -2486.7949219, 2033.1936035
2: -347.5960388, 1495.0695801, -549.7242432, 2323.1918945, -2670.7878418, 2044.7934570
3: -747.8043823, 1336.4666748, -1183.8676758, 2085.3574219, -2833.1616211, 2520.3344727
4: -598.7250366, 1399.0037842, -946.5105591, 2177.9230957, -2776.6477051, 2345.5144043

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B2_B1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7107829, upper bound: 1781.7178425
time: 0.51 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7107829, upper bound: 1781.7178425
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -288.9651489, 1172.4620361, -467.2495728, 1870.8992920, -2159.8645020, 1639.7116699
1: -465.5948486, 1297.6920166, -754.8636475, 2077.6809082, -2543.2758789, 2052.5556641
2: -347.5017395, 1494.6665039, -564.4549561, 2387.6782227, -2735.1799316, 2059.1215820
3: -747.6051636, 1336.0997314, -1215.8201904, 2143.1176758, -2890.7224121, 2551.9196777
4: -598.5634766, 1398.6253662, -971.8789062, 2237.6977539, -2836.2612305, 2370.5036621

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7110502, upper bound: 1781.7180200
time: 0.55 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7110502, upper bound: 1781.7180200
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -292.0651245, 1185.1374512, -345.9256897, 1397.2662354, -1689.3312988, 1531.0629883
1: -470.6372681, 1311.5079346, -557.6153564, 1548.8400879, -2019.4771729, 1869.1232910
2: -351.2402954, 1510.7570801, -416.4815063, 1781.9025879, -2133.1420898, 1927.2385254
3: -755.5812988, 1350.4630127, -897.8513184, 1593.7917480, -2349.3728027, 2248.3142090
4: -604.9828491, 1413.8060303, -717.3684082, 1667.8487549, -2272.8315430, 2131.1743164

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7160039, upper bound: 1781.7272486
time: 0.74 seconds

## Relational analysis of IS_A1_B2_B1_B1_B2

### Relational analysis result of IS_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7161104, upper bound: 1781.7268988
time: 0.50 seconds

## BFS IS instance: IS_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -292.0651245, 1185.1374512, -418.1317444, 1690.9853516, -1983.0505371, 1603.2691650
1: -470.6372681, 1311.5079346, -672.1157227, 1873.0083008, -2343.6455078, 1983.6235352
2: -351.2402954, 1510.7570801, -502.1312866, 2155.4829102, -2506.7231445, 2012.8883057
3: -755.5812988, 1350.4630127, -1081.2238770, 1928.4916992, -2684.0727539, 2431.6867676
4: -604.9828491, 1413.8060303, -866.1642456, 2016.1826172, -2621.1652832, 2279.9702148

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B2_B1

### Relational analysis result of IS_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7160039, upper bound: 1781.7272486
time: 0.56 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2

### Relational analysis result of IS_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7161104, upper bound: 1781.7268988
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -288.8876648, 1172.1311035, -396.1172180, 1589.1738281, -1878.0615234, 1568.2482910
1: -465.4729919, 1297.3371582, -639.6358643, 1764.6832275, -2230.1562500, 1936.9730225
2: -347.4084778, 1494.2482910, -478.4788208, 2027.5914307, -2375.0000000, 1972.7270508
3: -747.4121094, 1335.7172852, -1030.9722900, 1821.7375488, -2569.1491699, 2366.6894531
4: -598.4033203, 1398.2211914, -824.2117920, 1902.0307617, -2500.4340820, 2222.4323730

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7070154, upper bound: 1781.7070662
time: 0.57 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2

### Relational analysis result of IS_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7070154, upper bound: 1781.7070662
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -288.8876648, 1172.1311035, -483.0009155, 1938.7708740, -2227.6584473, 1655.1320801
1: -465.4729919, 1297.3371582, -779.0424194, 2151.9399414, -2617.4128418, 2076.3796387
2: -347.4084778, 1494.2482910, -582.8635864, 2473.3574219, -2820.7658691, 2077.1115723
3: -747.4121094, 1335.7172852, -1254.5689697, 2220.6840820, -2968.0957031, 2590.2861328
4: -598.4033203, 1398.2211914, -1004.4208984, 2317.7651367, -2916.1684570, 2402.6416016

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7070154, upper bound: 1781.7070662
time: 0.59 seconds

## Relational analysis of IS_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7070154, upper bound: 1781.7070662
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -366.0052185, 1479.9211426, -292.2078247, 1185.7346191, -1551.7396240, 1772.1287842
1: -589.1425781, 1638.8649902, -470.8653564, 1312.1569824, -1901.2995605, 2109.7304688
2: -440.0267944, 1886.3773193, -351.4121094, 1511.5142822, -1951.5410156, 2237.7895508
3: -946.3379517, 1689.0131836, -755.9409180, 1351.1541748, -2297.4919434, 2444.9541016
4: -758.7982178, 1765.4758301, -605.2770386, 1414.5281982, -2173.3264160, 2370.7526855

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7287093, upper bound: 1781.7151232
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7192287, upper bound: 1781.7110502
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2

### Relational analysis result of IS_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7192287, upper bound: 1781.7110502
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -460.4322510, 1844.2117920, -289.0429993, 1172.7789307, -1633.2111816, 2133.2546387
1: -744.2117920, 2047.9725342, -465.7215576, 1298.0424805, -2042.2542725, 2513.6940918
2: -556.5748291, 2353.6630859, -347.5960388, 1495.0695801, -2051.6442871, 2701.2590332
3: -1198.3090820, 2113.7905273, -747.8043823, 1336.4666748, -2534.7758789, 2861.5944824
4: -958.5090332, 2206.7712402, -598.7250366, 1399.0037842, -2357.5122070, 2805.4956055

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A2_A1

### Relational analysis result of IS_A2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7178425, upper bound: 1781.7107829
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_A2_A2

### Relational analysis result of IS_A2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7180200, upper bound: 1781.7110502
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -418.1317444, 1690.9853516, -292.0651245, 1185.1374512, -1603.2691650, 1983.0505371
1: -672.1157227, 1873.0083008, -470.6372681, 1311.5079346, -1983.6236572, 2343.6455078
2: -502.1312866, 2155.4829102, -351.2402954, 1510.7570801, -2012.8883057, 2506.7231445
3: -1081.2238770, 1928.4916992, -755.5812988, 1350.4630127, -2431.6867676, 2684.0727539
4: -866.1642456, 2016.1826172, -604.9828491, 1413.8060303, -2279.9702148, 2621.1652832

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_A1_A1

### Relational analysis result of IS_A2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7272483, upper bound: 1781.7160039
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_A1_A2

### Relational analysis result of IS_A2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7268988, upper bound: 1781.7161104
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -483.0009155, 1938.7708740, -288.8876648, 1172.1311035, -1655.1320801, 2227.6584473
1: -779.0424194, 2151.9399414, -465.4729919, 1297.3371582, -2076.3796387, 2617.4128418
2: -582.8635864, 2473.3574219, -347.4084778, 1494.2482910, -2077.1115723, 2820.7658691
3: -1254.5689697, 2220.6840820, -747.4121094, 1335.7172852, -2590.2861328, 2968.0957031
4: -1004.4208984, 2317.7651367, -598.4033203, 1398.2211914, -2402.6416016, 2916.1684570

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7070662, upper bound: 1781.7070154
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7070662, upper bound: 1781.7070154
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -380.0692139, 1536.6132812, -380.0692139, 1536.6132812, -1916.6823730, 1916.6823730
1: -611.8825684, 1701.3682861, -611.8825684, 1701.3682861, -2313.2507324, 2313.2507324
2: -457.1301575, 1958.7752686, -457.1301575, 1958.7752686, -2415.9055176, 2415.9055176
3: -982.6779175, 1754.2751465, -982.6779175, 1754.2751465, -2736.9521484, 2736.9523926
4: -788.3563232, 1833.5460205, -788.3563232, 1833.5460205, -2621.9020996, 2621.9020996

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7286327, upper bound: 1781.7153790
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7187430, upper bound: 1781.7103842
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -380.0692139, 1536.6132812, -433.4168091, 1752.8944092, -2132.9636230, 1970.0300293
1: -611.8825684, 1701.3682861, -696.7171021, 1941.2100830, -2553.0927734, 2398.0849609
2: -457.1301575, 1958.7752686, -520.5569458, 2234.4389648, -2691.5690918, 2479.3320312
3: -982.6779175, 1754.2751465, -1120.6684570, 1999.0828857, -2981.7607422, 2874.9433594
4: -788.3563232, 1833.5460205, -897.9285889, 2090.1264648, -2878.4824219, 2731.4746094

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7282178, upper bound: 1781.7334888
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7187430, upper bound: 1781.7110502
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -418.1317444, 1690.9853516, -407.6167908, 1648.4715576, -2066.6032715, 2098.6020508
1: -672.1157227, 1873.0083008, -655.8494263, 1824.8702393, -2496.9858398, 2528.8576660
2: -502.1312866, 2155.4829102, -489.9127502, 2100.9128418, -2603.0439453, 2645.3952637
3: -1081.2238770, 1928.4916992, -1053.5073242, 1881.4702148, -2962.6940918, 2981.9990234
4: -866.1642456, 2016.1826172, -844.6366577, 1966.9448242, -2833.1088867, 2860.8188477

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7283244, upper bound: 1781.7153161
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7283244, upper bound: 1781.7159918
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -483.0009155, 1938.7708740, -401.8054810, 1624.7401123, -2107.7409668, 2340.5761719
1: -779.0424194, 2151.9399414, -646.4685059, 1798.9355469, -2577.9780273, 2798.4084473
2: -582.8635864, 2473.3574219, -482.9635925, 2070.7175293, -2653.5808105, 2956.3208008
3: -1254.5689697, 2220.6840820, -1038.6633301, 1854.4073486, -3108.4907227, 3259.3474121
4: -1004.4208984, 2317.7651367, -832.7130127, 1938.4554443, -2942.8764648, 3150.4780273

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7059145, upper bound: 1781.7059145
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2

### Relational analysis result of IS_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7059145, upper bound: 1781.7070154
time: 0.58 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.59 seconds
IS_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -1781.7159641, upper bound: 1781.7269907
IS_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -1781.7160553, upper bound: 1781.7266019
IS_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -1781.7159641, upper bound: 1781.7269907
IS_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -1781.7160553, upper bound: 1781.7266019
IS_A1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -1781.7107829, upper bound: 1781.7178425
IS_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -1781.7107829, upper bound: 1781.7178425
IS_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -1781.7110502, upper bound: 1781.7180200
IS_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -1781.7110502, upper bound: 1781.7180200
IS_A1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -1781.7160039, upper bound: 1781.7272486
IS_A1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -1781.7161104, upper bound: 1781.7268988
IS_A1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -1781.7160039, upper bound: 1781.7272486
IS_A1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -1781.7161104, upper bound: 1781.7268988
IS_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -1781.7070154, upper bound: 1781.7070662
IS_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -1781.7070154, upper bound: 1781.7070662
IS_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -1781.7070154, upper bound: 1781.7070662
IS_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -1781.7070154, upper bound: 1781.7070662
IS_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -1781.7192287, upper bound: 1781.7110502
IS_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -1781.7192287, upper bound: 1781.7110502
IS_A2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -1781.7178425, upper bound: 1781.7107829
IS_A2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -1781.7180200, upper bound: 1781.7110502
IS_A2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -1781.7272483, upper bound: 1781.7160039
IS_A2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -1781.7268988, upper bound: 1781.7161104
IS_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -1781.7070662, upper bound: 1781.7070154
IS_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -1781.7070662, upper bound: 1781.7070154
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -1781.7286327, upper bound: 1781.7153790
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -1781.7187430, upper bound: 1781.7103842
IS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -1781.7282178, upper bound: 1781.7334888
IS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -1781.7187430, upper bound: 1781.7110502
IS_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -1781.7283244, upper bound: 1781.7153161
IS_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -1781.7283244, upper bound: 1781.7159918
IS_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -1781.7059145, upper bound: 1781.7059145
IS_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -1781.7059145, upper bound: 1781.7070154

## BFS IS instance: IS_A1_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -276.7585144, 1124.4237061, -354.4953918, 1433.8371582, -1710.5957031, 1478.9190674
1: -446.0162659, 1244.2011719, -570.7703247, 1587.6683350, -2033.6845703, 1814.9714355
2: -332.7811279, 1433.4194336, -426.1455078, 1827.7058105, -2160.4868164, 1859.5649414
3: -715.9465942, 1280.3992920, -917.2463379, 1635.1846924, -2351.1313477, 2197.6455078
4: -573.1133423, 1340.8247070, -734.4458618, 1710.2404785, -2283.3535156, 2075.2705078

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7359385, upper bound: 1781.7355396
time: 0.54 seconds

## Relational analysis of IS_A1_B1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7359385, upper bound: 1781.7356460
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -276.6843872, 1124.1207275, -364.5749512, 1473.3968506, -1750.0810547, 1488.6956787
1: -445.8954163, 1243.8666992, -587.4844360, 1631.6854248, -2077.5808105, 1831.3510742
2: -332.6913452, 1433.0345459, -438.4519958, 1878.2218018, -2210.9130859, 1871.4865723
3: -715.7567749, 1280.0474854, -943.5286255, 1681.8730469, -2397.6296387, 2223.5756836
4: -572.9594727, 1340.4621582, -755.4809570, 1758.7102051, -2331.6696777, 2095.9428711

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7360297, upper bound: 1781.7355396
time: 0.67 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7360297, upper bound: 1781.7356460
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -371.8542480, 1491.4521484, -354.4953918, 1433.8371582, -1805.6914062, 1845.9473877
1: -601.9755249, 1656.0629883, -570.7703247, 1587.6683350, -2189.6435547, 2226.8332520
2: -449.9300232, 1903.8698730, -426.1455078, 1827.7058105, -2277.6357422, 2330.0153809
3: -969.7329712, 1708.7253418, -917.2463379, 1635.1846924, -2604.9177246, 2625.9716797
4: -774.2708130, 1785.2263184, -734.4458618, 1710.2404785, -2484.5112305, 2519.6718750

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7150320, upper bound: 1781.7265610
time: 0.55 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7129105, upper bound: 1781.7252665
time: 0.59 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7146443, upper bound: 1781.7259975
time: 0.59 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7159193, upper bound: 1781.7247441
time: 0.52 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7159641, upper bound: 1781.7268719
time: 0.58 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7112971, upper bound: 1781.7224045
time: 0.61 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7157568, upper bound: 1781.7269907
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -371.7669678, 1491.1092529, -364.5749512, 1473.3968506, -1845.1636963, 1855.6842041
1: -601.8331909, 1655.6805420, -587.4844360, 1631.6854248, -2233.5185547, 2243.1650391
2: -449.8236389, 1903.4307861, -438.4519958, 1878.2218018, -2328.0454102, 2341.8828125
3: -969.5072021, 1708.3220215, -943.5286255, 1681.8730469, -2651.3801270, 2651.8505859
4: -774.0858765, 1784.8104248, -755.4809570, 1758.7102051, -2532.7958984, 2540.2915039

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7151232, upper bound: 1781.7263934
time: 0.56 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7147442, upper bound: 1781.7257586
time: 0.64 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7159848, upper bound: 1781.7244145
time: 0.59 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7129363, upper bound: 1781.7245588
time: 0.59 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7069996, upper bound: 1781.7139948
time: 0.57 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7095429, upper bound: 1781.7153336
time: 0.50 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -276.7585144, 1124.4237061, -454.6191711, 1819.8804932, -2096.6389160, 1579.0428467
1: -446.0162659, 1244.2011719, -735.1511841, 2021.0734863, -2467.0886230, 1979.3522949
2: -332.7811279, 1433.4194336, -549.7242432, 2323.1918945, -2655.9724121, 1983.1434326
3: -715.9465942, 1280.3992920, -1183.8676758, 2085.3574219, -2801.3039551, 2464.2670898
4: -573.1133423, 1340.8247070, -946.5105591, 2177.9230957, -2751.0363770, 2287.3349609

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7081475, upper bound: 1781.7168002
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7067194, upper bound: 1781.7174102
time: 0.48 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7107829, upper bound: 1781.7178425
time: 0.50 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -371.8542480, 1491.4521484, -454.6191711, 1819.8804932, -2191.7348633, 1946.0711670
1: -601.9755249, 1656.0629883, -735.1511841, 2021.0734863, -2623.0485840, 2391.2136230
2: -449.9300232, 1903.8698730, -549.7242432, 2323.1918945, -2773.1218262, 2453.5939941
3: -969.7329712, 1708.7253418, -1183.8676758, 2085.3574219, -3054.7324219, 2892.4279785
4: -774.2708130, 1785.2263184, -946.5105591, 2177.9230957, -2952.1938477, 2731.7363281

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7067194, upper bound: 1781.7174102
time: 0.58 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7107829, upper bound: 1781.7178425
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -276.6843872, 1124.1207275, -467.2495728, 1870.8992920, -2147.5837402, 1591.3703613
1: -445.8954163, 1243.8666992, -754.8636475, 2077.6809082, -2523.5764160, 1998.7302246
2: -332.6913452, 1433.0345459, -564.4549561, 2387.6782227, -2720.3696289, 1997.4895020
3: -715.7567749, 1280.0474854, -1215.8201904, 2143.1176758, -2858.8745117, 2495.8674316
4: -572.9594727, 1340.4621582, -971.8789062, 2237.6977539, -2810.6569824, 2312.3403320

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7080799, upper bound: 1781.7153270
time: 0.55 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7099713, upper bound: 1781.7175595
time: 0.57 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7056321, upper bound: 1781.7057343
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -371.7669678, 1491.1092529, -467.2495728, 1870.8992920, -2242.6662598, 1958.3588867
1: -601.8331909, 1655.6805420, -754.8636475, 2077.6809082, -2679.5139160, 2410.5441895
2: -449.8236389, 1903.4307861, -564.4549561, 2387.6782227, -2837.5019531, 2467.8854980
3: -969.5072021, 1708.3220215, -1215.8201904, 2143.1176758, -3112.1804199, 2923.9143066
4: -774.0858765, 1784.8104248, -971.8789062, 2237.6977539, -3011.7834473, 2756.6889648

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7110502, upper bound: 1781.7174801
time: 0.55 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7085933, upper bound: 1781.7173234
time: 0.54 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7099713, upper bound: 1781.7175595
time: 0.55 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7056321, upper bound: 1781.7057343
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -292.0651245, 1185.1374512, -331.0666504, 1336.9929199, -1629.0581055, 1516.2041016
1: -470.6372681, 1311.5079346, -533.9237671, 1482.3168945, -1952.9541016, 1845.4315186
2: -351.2402954, 1510.7570801, -398.7275696, 1705.2055664, -2056.4458008, 1909.4846191
3: -755.5812988, 1350.4630127, -860.2004395, 1524.3692627, -2279.9501953, 2210.6635742
4: -604.9828491, 1413.8060303, -686.3527222, 1595.9561768, -2200.9389648, 2100.1586914

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1_B1_B1_A1

### Relational analysis result of IS_A1_B2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7150718, upper bound: 1781.7268220
time: 0.56 seconds

## Relational analysis of IS_A1_B2_B1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B1_B1_B1_A1

### Relational analysis result of IS_A1_B2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7160039, upper bound: 1781.7272486
time: 0.51 seconds

## Relational analysis of IS_A1_B2_B1_B1_B1_A2

### Relational analysis result of IS_A1_B2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7160039, upper bound: 1781.7272486
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -291.9880676, 1184.8243408, -347.6588440, 1401.9628906, -1693.9509277, 1532.4829102
1: -470.5116577, 1311.1612549, -561.1382446, 1554.5028076, -2025.0144043, 1872.2995605
2: -351.1470032, 1510.3587646, -418.7875061, 1788.1040039, -2139.2507324, 1929.1461182
3: -755.3839111, 1350.0996094, -903.4758301, 1599.8117676, -2355.1958008, 2253.5751953
4: -604.8228149, 1413.4316406, -720.4724731, 1674.6651611, -2279.4880371, 2133.9035645

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1_B1_B2_A1

### Relational analysis result of IS_A1_B2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7151783, upper bound: 1781.7267860
time: 0.54 seconds

## Relational analysis of IS_A1_B2_B1_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B1_B1_B2_A1

### Relational analysis result of IS_A1_B2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7161104, upper bound: 1781.7270115
time: 0.55 seconds

## Relational analysis of IS_A1_B2_B1_B1_B2_A2

### Relational analysis result of IS_A1_B2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7161104, upper bound: 1781.7270115
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -292.0651245, 1185.1374512, -403.6436768, 1632.2117920, -1924.2768555, 1588.7811279
1: -470.6372681, 1311.5079346, -648.9354858, 1808.0749512, -2278.7119141, 1960.4433594
2: -351.2402954, 1510.7570801, -484.7603149, 2080.5888672, -2431.8291016, 1995.5173340
3: -755.5812988, 1350.4630127, -1044.3218994, 1860.5911865, -2616.1721191, 2394.7849121
4: -604.9828491, 1413.8060303, -835.8314819, 1945.9047852, -2550.8876953, 2249.6374512

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7150718, upper bound: 1781.7268220
time: 0.57 seconds

## Relational analysis of IS_A1_B2_B1_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7160039, upper bound: 1781.7272486
time: 0.57 seconds

## Relational analysis of IS_A1_B2_B1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7160039, upper bound: 1781.7272486
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -291.9880676, 1184.8243408, -416.2760010, 1681.2331543, -1973.2211914, 1601.1000977
1: -470.5116577, 1311.1612549, -669.5971680, 1862.8013916, -2333.3125000, 1980.7584229
2: -351.1470032, 1510.3587646, -499.9979248, 2143.3747559, -2494.5217285, 2010.3564453
3: -755.3839111, 1350.0996094, -1077.4113770, 1918.1418457, -2673.5258789, 2427.5109863
4: -604.8228149, 1413.4316406, -861.9510498, 2005.7935791, -2610.6164551, 2275.3828125

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7151783, upper bound: 1781.7266903
time: 0.56 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7161104, upper bound: 1781.7268988
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7161104, upper bound: 1781.7268988
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -276.6474304, 1123.9506836, -396.1172180, 1589.1738281, -1865.8212891, 1520.0678711
1: -445.8388367, 1243.6837158, -639.6358643, 1764.6832275, -2210.5219727, 1883.3195801
2: -332.6482849, 1432.8239746, -478.4788208, 2027.5914307, -2360.2395020, 1911.3024902
3: -715.6668701, 1279.8508301, -1030.9722900, 1821.7375488, -2537.4042969, 2310.8227539
4: -572.8852539, 1340.2568359, -824.2117920, 1902.0307617, -2474.9157715, 2164.4687500

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7044850, upper bound: 1781.7032804
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7058238, upper bound: 1781.7058238
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -371.6766663, 1490.7374268, -396.1172180, 1589.1738281, -1960.8504639, 1886.8546143
1: -601.6890869, 1655.2745361, -639.6358643, 1764.6832275, -2366.3723145, 2294.9104004
2: -449.7159729, 1902.9573975, -478.4788208, 2027.5914307, -2477.3073730, 2381.4362793
3: -969.2743530, 1707.9077148, -1030.9722900, 1821.7375488, -2790.5734863, 2738.8798828
4: -773.9030151, 1784.3653564, -824.2117920, 1902.0307617, -2675.9335938, 2608.5771484

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7044850, upper bound: 1781.7032804
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7058238, upper bound: 1781.7058238
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -276.6474304, 1123.9506836, -483.0009155, 1938.7708740, -2215.4174805, 1606.9516602
1: -445.8388367, 1243.6837158, -779.0424194, 2151.9399414, -2597.7788086, 2022.7260742
2: -332.6482849, 1432.8239746, -582.8635864, 2473.3574219, -2806.0051270, 2015.6875000
3: -715.6668701, 1279.8508301, -1254.5689697, 2220.6840820, -2936.3510742, 2534.4196777
4: -572.8852539, 1340.2568359, -1004.4208984, 2317.7651367, -2890.6503906, 2344.6777344

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7054382, upper bound: 1781.7053345
time: 0.76 seconds

## Relational analysis of IS_A1_B2_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7056665, upper bound: 1781.7058196
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -371.6766663, 1490.7374268, -483.0009155, 1938.7708740, -2310.4472656, 1973.7382812
1: -601.6890869, 1655.2745361, -779.0424194, 2151.9399414, -2753.6289062, 2434.3168945
2: -449.7159729, 1902.9573975, -582.8635864, 2473.3574219, -2923.0732422, 2485.8210449
3: -969.2743530, 1707.9077148, -1254.5689697, 2220.6840820, -3187.5329590, 2962.4765625
4: -773.9030151, 1784.3653564, -1004.4208984, 2317.7651367, -3091.6682129, 2788.7861328

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7054382, upper bound: 1781.7053345
time: 0.51 seconds

## Relational analysis of IS_A1_B2_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7056665, upper bound: 1781.7058196
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -366.0052185, 1479.9211426, -276.7585144, 1124.4237061, -1490.4287109, 1756.6794434
1: -589.1425781, 1638.8649902, -446.0162659, 1244.2011719, -1833.3437500, 2084.8808594
2: -440.0267944, 1886.3773193, -332.7811279, 1433.4194336, -1873.4461670, 2219.1584473
3: -946.3379517, 1689.0131836, -715.9465942, 1280.3992920, -2226.7373047, 2404.9597168
4: -758.7982178, 1765.4758301, -573.1133423, 1340.8247070, -2099.6228027, 2338.5888672

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7287093, upper bound: 1781.7151232
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7269907, upper bound: 1781.7159641
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7266019, upper bound: 1781.7160553
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -366.0052185, 1479.9211426, -371.8542480, 1491.4521484, -1857.4571533, 1851.7753906
1: -589.1425781, 1638.8649902, -601.9755249, 1656.0629883, -2245.2055664, 2240.8405762
2: -440.0267944, 1886.3773193, -449.9300232, 1903.8698730, -2343.8967285, 2336.3073730
3: -946.3379517, 1689.0131836, -969.7329712, 1708.7253418, -2655.0632324, 2657.9223633
4: -758.7982178, 1765.4758301, -774.2708130, 1785.2263184, -2544.0241699, 2539.7465820

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A1_B2_B1

### Relational analysis result of IS_A2_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7287093, upper bound: 1781.7151232
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7269907, upper bound: 1781.7159641
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7266019, upper bound: 1781.7160553
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -451.2770996, 1805.6795654, -289.0429993, 1172.7789307, -1624.0560303, 2094.7221680
1: -729.7014160, 2005.7186279, -465.7215576, 1298.0424805, -2027.7438965, 2471.4401855
2: -545.6817017, 2305.0024414, -347.5960388, 1495.0695801, -2040.7512207, 2652.5983887
3: -1175.3560791, 2069.5727539, -747.8043823, 1336.4666748, -2511.8227539, 2817.3767090
4: -939.5781250, 2161.1323242, -598.7250366, 1399.0037842, -2338.5820312, 2759.8569336

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A2_A1_B1

### Relational analysis result of IS_A2_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7168002, upper bound: 1781.7081475
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_A2_A1_B1

### Relational analysis result of IS_A2_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7178425, upper bound: 1781.7107829
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_A2_A1_B2

### Relational analysis result of IS_A2_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7178425, upper bound: 1781.7107829
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -464.3292236, 1858.6447754, -288.9651489, 1172.4620361, -1636.7912598, 2147.6098633
1: -750.1074829, 2064.3579102, -465.5948486, 1297.6920166, -2047.7993164, 2529.9526367
2: -560.9035034, 2371.9299316, -347.5017395, 1494.6665039, -2055.5700684, 2719.4316406
3: -1208.3598633, 2129.5061035, -747.6051636, 1336.0997314, -2544.4594727, 2877.1108398
4: -965.7791138, 2223.2324219, -598.5634766, 1398.6253662, -2364.4038086, 2821.7954102

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A2_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7153270, upper bound: 1781.7080799
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_A2_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7180200, upper bound: 1781.7110502
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_A2_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7180200, upper bound: 1781.7110502
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -403.6436768, 1632.2117920, -292.0651245, 1185.1374512, -1588.7811279, 1924.2768555
1: -648.9354858, 1808.0749512, -470.6372681, 1311.5079346, -1960.4433594, 2278.7119141
2: -484.7603149, 2080.5888672, -351.2402954, 1510.7570801, -1995.5173340, 2431.8291016
3: -1044.3218994, 1860.5911865, -755.5812988, 1350.4630127, -2394.7846680, 2616.1721191
4: -835.8314819, 1945.9047852, -604.9828491, 1413.8060303, -2249.6374512, 2550.8876953

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A1_A1_B1

### Relational analysis result of IS_A2_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7268220, upper bound: 1781.7150718
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A2_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_A1_A1_B1

### Relational analysis result of IS_A2_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7272483, upper bound: 1781.7160039
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_A1_A1_B2

### Relational analysis result of IS_A2_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7272483, upper bound: 1781.7160039
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -416.2760010, 1681.2331543, -291.9880676, 1184.8243408, -1601.1002197, 1973.2211914
1: -669.5971680, 1862.8013916, -470.5116577, 1311.1612549, -1980.7584229, 2333.3127441
2: -499.9979248, 2143.3747559, -351.1470032, 1510.3587646, -2010.3564453, 2494.5214844
3: -1077.4113770, 1918.1418457, -755.3839111, 1350.0996094, -2427.5109863, 2673.5258789
4: -861.9510498, 2005.7935791, -604.8228149, 1413.4316406, -2275.3828125, 2610.6164551

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A1_A2_B1

### Relational analysis result of IS_A2_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7266903, upper bound: 1781.7151783
time: 0.49 seconds

## Relational analysis of IS_A2_B1_A2_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_A1_A2_B1

### Relational analysis result of IS_A2_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7268988, upper bound: 1781.7161104
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_A1_A2_B2

### Relational analysis result of IS_A2_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7268988, upper bound: 1781.7161104
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -483.0009155, 1938.7708740, -276.6474304, 1123.9506836, -1606.9516602, 2215.4174805
1: -779.0424194, 2151.9399414, -445.8388367, 1243.6837158, -2022.7260742, 2597.7788086
2: -582.8635864, 2473.3574219, -332.6482849, 1432.8239746, -2015.6875000, 2806.0051270
3: -1254.5689697, 2220.6840820, -715.6668701, 1279.8508301, -2534.4196777, 2936.3510742
4: -1004.4208984, 2317.7651367, -572.8852539, 1340.2568359, -2344.6777344, 2890.6503906

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7053345, upper bound: 1781.7054382
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7058196, upper bound: 1781.7056665
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -483.0009155, 1938.7708740, -371.6766663, 1490.7374268, -1973.7382812, 2310.4472656
1: -779.0424194, 2151.9399414, -601.6890869, 1655.2745361, -2434.3168945, 2753.6289062
2: -582.8635864, 2473.3574219, -449.7159729, 1902.9573975, -2485.8210449, 2923.0732422
3: -1254.5689697, 2220.6840820, -969.2743530, 1707.9077148, -2962.4765625, 3187.5329590
4: -1004.4208984, 2317.7651367, -773.9030151, 1784.3653564, -2788.7861328, 3091.6682129

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7053345, upper bound: 1781.7054382
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7058196, upper bound: 1781.7056665
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -366.0052185, 1479.9211426, -380.0692139, 1536.6132812, -1902.6182861, 1859.9901123
1: -589.1425781, 1638.8649902, -611.8825684, 1701.3682861, -2290.5104980, 2250.7475586
2: -440.0267944, 1886.3773193, -457.1301575, 1958.7752686, -2398.8020020, 2343.5075684
3: -946.3379517, 1689.0131836, -982.6779175, 1754.2751465, -2700.6127930, 2671.6909180
4: -758.7982178, 1765.4758301, -788.3563232, 1833.5460205, -2592.3442383, 2553.8315430

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7232127, upper bound: 1781.7228391
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7232127, upper bound: 1781.7228391
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -460.4322510, 1844.2117920, -375.0501099, 1515.9726562, -1976.4049072, 2219.2619629
1: -744.2117920, 2047.9725342, -603.7593384, 1678.8731689, -2423.0849609, 2651.7319336
2: -556.5748291, 2353.6630859, -451.1304321, 1932.5198975, -2489.0942383, 2804.7929688
3: -1198.3090820, 2113.7905273, -969.8624268, 1730.7391357, -2927.5832520, 3083.6528320
4: -958.5090332, 2206.7712402, -778.0573730, 1808.7648926, -2767.2734375, 2984.8281250

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7218264, upper bound: 1781.7217367
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7220039, upper bound: 1781.7219876
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -380.0692139, 1536.6132812, -418.1317444, 1690.9853516, -2071.0544434, 1954.7449951
1: -611.8825684, 1701.3682861, -672.1157227, 1873.0083008, -2484.8908691, 2373.4833984
2: -457.1301575, 1958.7752686, -502.1312866, 2155.4829102, -2612.6130371, 2460.9064941
3: -982.6779175, 1754.2751465, -1081.2238770, 1928.4916992, -2911.1694336, 2835.4982910
4: -788.3563232, 1833.5460205, -866.1642456, 2016.1826172, -2804.5383301, 2699.7102051

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_B1_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7269576, upper bound: 1781.7313763
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_B2

### Relational analysis result of IS_A2_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7270641, upper bound: 1781.7308827
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -375.0501099, 1515.9726562, -483.0009155, 1938.7708740, -2313.8208008, 1998.9736328
1: -603.7593384, 1678.8731689, -779.0424194, 2151.9399414, -2755.6992188, 2457.9155273
2: -451.1304321, 1932.5198975, -582.8635864, 2473.3574219, -2924.4870605, 2515.3835449
3: -969.8624268, 1730.7391357, -1254.5689697, 2220.6840820, -3190.5463867, 2984.9260254
4: -778.0573730, 1808.7648926, -1004.4208984, 2317.7651367, -3095.8225098, 2813.1857910

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7168869, upper bound: 1781.7079123
time: 1.86 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7191779, upper bound: 1781.7110502
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7191779, upper bound: 1781.7110502
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -418.1317444, 1690.9853516, -380.0692139, 1536.6132812, -1954.7449951, 2071.0544434
1: -672.1157227, 1873.0083008, -611.8825684, 1701.3682861, -2373.4833984, 2484.8908691
2: -502.1312866, 2155.4829102, -457.1301575, 1958.7752686, -2460.9064941, 2612.6130371
3: -1081.2238770, 1928.4916992, -982.6779175, 1754.2751465, -2835.4982910, 2911.1694336
4: -866.1642456, 2016.1826172, -788.3563232, 1833.5460205, -2699.7102051, 2804.5383301

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7263118, upper bound: 1781.7152136
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7258625, upper bound: 1781.7153161
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -418.1317444, 1690.9853516, -433.4168091, 1752.8944092, -2171.0261230, 2124.4020996
1: -672.1157227, 1873.0083008, -696.7171021, 1941.2100830, -2613.3256836, 2569.7253418
2: -502.1312866, 2155.4829102, -520.5569458, 2234.4389648, -2736.5700684, 2676.0393066
3: -1081.2238770, 1928.4916992, -1120.6684570, 1999.0828857, -3080.3066406, 3049.1601562
4: -866.1642456, 2016.1826172, -897.9285889, 2090.1264648, -2956.2907715, 2914.1105957

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7283244, upper bound: 1781.7159918
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7283244, upper bound: 1781.7159918
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -483.0009155, 1938.7708740, -375.0501099, 1515.9726562, -1998.9736328, 2313.8208008
1: -779.0424194, 2151.9399414, -603.7593384, 1678.8731689, -2457.9155273, 2755.6992188
2: -582.8635864, 2473.3574219, -451.1304321, 1932.5198975, -2515.3830566, 2924.4870605
3: -1254.5689697, 2220.6840820, -969.8624268, 1730.7391357, -2984.9260254, 3190.5463867
4: -1004.4208984, 2317.7651367, -778.0573730, 1808.7648926, -2813.1857910, 3095.8225098

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7059145, upper bound: 1781.7059145
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7059145, upper bound: 1781.7059145
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -483.0009155, 1938.7708740, -427.9795532, 1730.7586670, -2213.7595215, 2366.7502441
1: -779.0424194, 2151.9399414, -687.9745483, 1916.9886475, -2696.0310059, 2839.9145508
2: -582.8635864, 2473.3574219, -514.0354004, 2206.2893066, -2789.1528320, 2987.3925781
3: -1254.5689697, 2220.6840820, -1106.8381348, 1973.7304688, -3227.5563965, 3327.5222168
4: -1004.4208984, 2317.7651367, -886.7081909, 2063.5612793, -3067.9821777, 3204.4733887

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7059145, upper bound: 1781.7070154
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7059145, upper bound: 1781.7070154
time: 0.60 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.65 seconds
IS_A1_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7359385, upper bound: 1781.7355396
IS_A1_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7359385, upper bound: 1781.7356460
IS_A1_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7360297, upper bound: 1781.7355396
IS_A1_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7360297, upper bound: 1781.7356460
IS_A1_B1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7112971, upper bound: 1781.7224045
IS_A1_B1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7157568, upper bound: 1781.7269907
IS_A1_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7069996, upper bound: 1781.7139948
IS_A1_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7095429, upper bound: 1781.7153336
IS_A1_B1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7067194, upper bound: 1781.7174102
IS_A1_B1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7107829, upper bound: 1781.7178425
IS_A1_B1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7067194, upper bound: 1781.7174102
IS_A1_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7107829, upper bound: 1781.7178425
IS_A1_B1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7099713, upper bound: 1781.7175595
IS_A1_B1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7056321, upper bound: 1781.7057343
IS_A1_B1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7099713, upper bound: 1781.7175595
IS_A1_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7056321, upper bound: 1781.7057343
IS_A1_B2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7160039, upper bound: 1781.7272486
IS_A1_B2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7160039, upper bound: 1781.7272486
IS_A1_B2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7161104, upper bound: 1781.7270115
IS_A1_B2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7161104, upper bound: 1781.7270115
IS_A1_B2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7160039, upper bound: 1781.7272486
IS_A1_B2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7160039, upper bound: 1781.7272486
IS_A1_B2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7161104, upper bound: 1781.7268988
IS_A1_B2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7161104, upper bound: 1781.7268988
IS_A1_B2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7044850, upper bound: 1781.7032804
IS_A1_B2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7058238, upper bound: 1781.7058238
IS_A1_B2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7044850, upper bound: 1781.7032804
IS_A1_B2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7058238, upper bound: 1781.7058238
IS_A1_B2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7054382, upper bound: 1781.7053345
IS_A1_B2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7056665, upper bound: 1781.7058196
IS_A1_B2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7054382, upper bound: 1781.7053345
IS_A1_B2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7056665, upper bound: 1781.7058196
IS_A2_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7269907, upper bound: 1781.7159641
IS_A2_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7266019, upper bound: 1781.7160553
IS_A2_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7269907, upper bound: 1781.7159641
IS_A2_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7266019, upper bound: 1781.7160553
IS_A2_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7178425, upper bound: 1781.7107829
IS_A2_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7178425, upper bound: 1781.7107829
IS_A2_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7180200, upper bound: 1781.7110502
IS_A2_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7180200, upper bound: 1781.7110502
IS_A2_B1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7272483, upper bound: 1781.7160039
IS_A2_B1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7272483, upper bound: 1781.7160039
IS_A2_B1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7268988, upper bound: 1781.7161104
IS_A2_B1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7268988, upper bound: 1781.7161104
IS_A2_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7053345, upper bound: 1781.7054382
IS_A2_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7058196, upper bound: 1781.7056665
IS_A2_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7053345, upper bound: 1781.7054382
IS_A2_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7058196, upper bound: 1781.7056665
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7232127, upper bound: 1781.7228391
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7232127, upper bound: 1781.7228391
IS_A2_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7218264, upper bound: 1781.7217367
IS_A2_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7220039, upper bound: 1781.7219876
IS_A2_B2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7269576, upper bound: 1781.7313763
IS_A2_B2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7270641, upper bound: 1781.7308827
IS_A2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7191779, upper bound: 1781.7110502
IS_A2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7191779, upper bound: 1781.7110502
IS_A2_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7263118, upper bound: 1781.7152136
IS_A2_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7258625, upper bound: 1781.7153161
IS_A2_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7283244, upper bound: 1781.7159918
IS_A2_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7283244, upper bound: 1781.7159918
IS_A2_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7059145, upper bound: 1781.7059145
IS_A2_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7059145, upper bound: 1781.7059145
IS_A2_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7059145, upper bound: 1781.7070154
IS_A2_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1781.7059145, upper bound: 1781.7070154

## BFS IS instance: IS_A1_B1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -262.0270691, 1064.5992432, -354.4953918, 1433.8371582, -1695.8642578, 1419.0943604
1: -422.6104431, 1178.1564941, -570.7703247, 1587.6683350, -2010.2788086, 1748.9267578
2: -315.2193298, 1357.1788330, -426.1455078, 1827.7058105, -2142.9250488, 1783.3243408
3: -678.7393188, 1211.5223389, -917.2463379, 1635.1846924, -2313.9240723, 2128.7685547
4: -542.2911377, 1269.4912109, -734.4458618, 1710.2404785, -2252.5314941, 2003.9370117

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7340729, upper bound: 1781.7349264
time: 0.64 seconds

## Relational analysis of IS_A1_B1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7351206, upper bound: 1781.7349489
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -277.5161743, 1124.4520264, -354.4953918, 1433.8371582, -1711.3532715, 1478.9472656
1: -448.0932617, 1244.2868652, -570.7703247, 1587.6683350, -2035.7614746, 1815.0571289
2: -334.0101624, 1433.9173584, -426.1455078, 1827.7058105, -2161.7160645, 1860.0628662
3: -718.8878174, 1281.1759033, -917.2463379, 1635.1846924, -2354.0725098, 2198.4221191
4: -574.2120361, 1342.3636475, -734.4458618, 1710.2404785, -2284.4523926, 2076.8095703

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7340729, upper bound: 1781.7350328
time: 0.55 seconds

## Relational analysis of IS_A1_B1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7351206, upper bound: 1781.7350553
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -262.0270691, 1064.5992432, -364.5749512, 1473.3968506, -1735.4238281, 1429.1741943
1: -422.6104431, 1178.1564941, -587.4844360, 1631.6854248, -2054.2958984, 1765.6408691
2: -315.2193298, 1357.1788330, -438.4519958, 1878.2218018, -2193.4409180, 1795.6308594
3: -678.7393188, 1211.5223389, -943.5286255, 1681.8730469, -2360.6118164, 2155.0510254
4: -542.2911377, 1269.4912109, -755.4809570, 1758.7102051, -2301.0014648, 2024.9720459

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_A1

### Relational analysis result of IS_A1_B1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7347568, upper bound: 1781.7322319
time: 0.57 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_A2

### Relational analysis result of IS_A1_B1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7357313, upper bound: 1781.7352913
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -277.5161743, 1124.4520264, -364.5749512, 1473.3968506, -1750.9129639, 1489.0269775
1: -448.0932617, 1244.2868652, -587.4844360, 1631.6854248, -2079.7788086, 1831.7712402
2: -334.0101624, 1433.9173584, -438.4519958, 1878.2218018, -2212.2319336, 1872.3693848
3: -718.8878174, 1281.1759033, -943.5286255, 1681.8730469, -2400.7602539, 2224.7038574
4: -574.2120361, 1342.3636475, -755.4809570, 1758.7102051, -2332.9223633, 2097.8447266

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_A1

### Relational analysis result of IS_A1_B1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7351276, upper bound: 1781.7336029
time: 0.63 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_A1

### Relational analysis result of IS_A1_B1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7333896, upper bound: 1781.7326035
time: 0.61 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_A2

### Relational analysis result of IS_A1_B1_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7333798, upper bound: 1781.7332625
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -370.5192566, 1485.9195557, -281.0463257, 1139.0540771, -1509.5732422, 1766.9656982
1: -599.8128052, 1649.9384766, -451.2229919, 1257.7279053, -1857.5407715, 2101.1611328
2: -448.3293457, 1896.8321533, -337.6435852, 1451.8374023, -1900.1667480, 2234.4755859
3: -966.2703247, 1702.4954834, -723.3122559, 1301.2858887, -2267.5559082, 2425.8071289
4: -771.5334473, 1778.6892090, -583.3668213, 1361.1368408, -2132.6701660, 2362.0559082

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7101300, upper bound: 1781.7218717
time: 0.58 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7088185, upper bound: 1781.7217249
time: 0.67 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7112971, upper bound: 1781.7215551
time: 0.73 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7112971, upper bound: 1781.7224045
time: 0.53 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7105110, upper bound: 1781.7217897
time: 0.66 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7111708, upper bound: 1781.7221647
time: 0.56 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 0
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 0
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 34
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 44
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 37

Time for candidate selection: 13.90 seconds

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7112971, upper bound: 1781.7224045
time: 0.64 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7112971, upper bound: 1781.7224045
time: 0.68 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7109973, upper bound: 1781.7223404
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -371.8542480, 1491.4521484, -351.1799927, 1420.6140137, -1792.4682617, 1842.6320801
1: -601.9755249, 1656.0629883, -565.3056030, 1573.0965576, -2175.0720215, 2221.3679199
2: -449.9300232, 1903.8698730, -422.0693054, 1810.8201904, -2260.7502441, 2325.9392090
3: -969.7329712, 1708.7253418, -908.6416016, 1620.0386963, -2589.7717285, 2617.3669434
4: -774.2708130, 1785.2263184, -727.4542236, 1694.3819580, -2468.6523438, 2512.6801758

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7148248, upper bound: 1781.7265610
time: 0.54 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7129046, upper bound: 1781.7252665
time: 0.55 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7143148, upper bound: 1781.7259975
time: 0.55 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7157568, upper bound: 1781.7247441
time: 0.57 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_B1

### Relational analysis result of IS_A1_B1_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7157568, upper bound: 1781.7268719
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_B1

### Relational analysis result of IS_A1_B1_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7149748, upper bound: 1781.7264238
time: 0.72 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7156306, upper bound: 1781.7261162
time: 0.50 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 0
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 0
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 34
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 44
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 9
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 1

Time for candidate selection: 14.00 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7136280, upper bound: 1781.7250829
time: 0.55 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7151107, upper bound: 1781.7254262
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -369.5518188, 1482.2154541, -364.5749512, 1473.3968506, -1842.9483643, 1846.7904053
1: -598.2822266, 1645.8143311, -587.4844360, 1631.6854248, -2229.9677734, 2233.2988281
2: -447.1547546, 1892.0567627, -438.4519958, 1878.2218018, -2325.3764648, 2330.5087891
3: -963.7833862, 1698.1500244, -943.5286255, 1681.8730469, -2645.6557617, 2641.6784668
4: -769.4749146, 1774.1794434, -755.4809570, 1758.7102051, -2528.1850586, 2529.6604004

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7069996, upper bound: 1781.7139948
time: 0.59 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7069996, upper bound: 1781.7139948
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -369.1216736, 1480.2462158, -364.4251099, 1472.7863770, -1841.9079590, 1844.6713867
1: -597.5000610, 1643.6959229, -587.2423706, 1631.0111084, -2228.5112305, 2230.9377441
2: -446.6363525, 1889.5932617, -438.2720642, 1877.4421387, -2324.0786133, 2327.8652344
3: -962.5603638, 1695.8374023, -943.1404419, 1681.1806641, -2643.7409668, 2638.9777832
4: -768.5286865, 1771.7824707, -755.1713257, 1757.9812012, -2526.5095215, 2526.9531250

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7095429, upper bound: 1781.7153336
time: 0.57 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7095429, upper bound: 1781.7153336
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -275.0691528, 1117.4814453, -388.5863037, 1554.9687500, -1830.0378418, 1506.0676270
1: -443.3122559, 1236.5261230, -627.7641602, 1725.4652100, -2168.7773438, 1864.2902832
2: -330.7697754, 1424.5583496, -469.8488464, 1985.2552490, -2316.0249023, 1894.4071045
3: -711.5732422, 1272.6407471, -1010.7808838, 1783.5228271, -2495.0957031, 2283.2915039
4: -569.6405640, 1332.6560059, -810.1556396, 1862.3239746, -2431.9645996, 2142.8115234

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7265630, upper bound: 1781.7261271
time: 0.61 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7265630, upper bound: 1781.7264829
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -276.7585144, 1124.4237061, -447.9162292, 1792.8598633, -2069.6181641, 1572.3399658
1: -446.0162659, 1244.2011719, -724.0599365, 1991.0988770, -2437.1145020, 1968.2611084
2: -332.7811279, 1433.4194336, -541.7187500, 2288.5012207, -2621.2819824, 1975.1381836
3: -715.9465942, 1280.3992920, -1165.4941406, 2056.1088867, -2772.0554199, 2445.8935547
4: -573.1133423, 1340.8247070, -932.7473145, 2147.0283203, -2720.1413574, 2273.5715332

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7307373, upper bound: 1781.7265313
time: 0.54 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7307373, upper bound: 1781.7268866
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -370.5192566, 1485.9195557, -388.5863037, 1554.9687500, -1925.4879150, 1874.5057373
1: -599.8128052, 1649.9384766, -627.7641602, 1725.4652100, -2325.2780762, 2277.7019043
2: -448.3293457, 1896.8321533, -469.8488464, 1985.2552490, -2433.5842285, 2366.6804199
3: -966.2703247, 1702.4954834, -1010.7808838, 1783.5228271, -2749.2631836, 2711.8408203
4: -771.5334473, 1778.6892090, -810.1556396, 1862.3239746, -2633.8574219, 2588.8447266

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7042596, upper bound: 1781.7172070
time: 0.64 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7067194, upper bound: 1781.7170674
time: 0.57 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7065931, upper bound: 1781.7171112
time: 0.54 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7054157, upper bound: 1781.7165642
time: 0.54 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7058815, upper bound: 1781.7168631
time: 0.55 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7066218, upper bound: 1781.7173473
time: 0.52 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 0
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 0
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 37

Time for candidate selection: 13.63 seconds

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 47

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7067194, upper bound: 1781.7174102
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7067194, upper bound: 1781.7174102
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -371.8542480, 1491.4521484, -447.9162292, 1792.8598633, -2164.7138672, 1939.3684082
1: -601.9755249, 1656.0629883, -724.0599365, 1991.0988770, -2593.0744629, 2380.1230469
2: -449.9300232, 1903.8698730, -541.7187500, 2288.5012207, -2738.4311523, 2445.5886230
3: -969.7329712, 1708.7253418, -1165.4941406, 2056.1088867, -3025.4799805, 2874.2194824
4: -774.2708130, 1785.2263184, -932.7473145, 2147.0283203, -2921.2990723, 2717.9729004

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7107829, upper bound: 1781.7172172
time: 0.56 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7083260, upper bound: 1781.7174018
time: 0.57 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7095026, upper bound: 1781.7171340
time: 0.57 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7081475, upper bound: 1781.7168002
time: 0.57 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7098792, upper bound: 1781.7173820
time: 0.54 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7104165, upper bound: 1781.7177987
time: 0.59 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 0
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 0
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 44
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 34
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 9
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 25

Time for candidate selection: 13.48 seconds

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 47

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7107829, upper bound: 1781.7178425
time: 0.59 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_B2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7106970, upper bound: 1781.7178425
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -276.6843872, 1124.1207275, -464.6970520, 1860.5928955, -2137.2773438, 1588.8177490
1: -445.8954163, 1243.8666992, -750.7463379, 2066.2927246, -2512.1882324, 1994.6130371
2: -332.6913452, 1433.0345459, -561.3922119, 2374.5144043, -2707.2058105, 1994.4267578
3: -715.7567749, 1280.0474854, -1209.1827393, 2131.3957520, -2847.1525879, 2489.2302246
4: -572.9594727, 1340.4621582, -966.5844116, 2225.3842773, -2798.3435059, 2307.0461426

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B2_A1_B1_B1

### Relational analysis result of IS_A1_B1_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7281055, upper bound: 1781.7216503
time: 0.64 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7282788, upper bound: 1781.7232797
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -276.5044556, 1123.4060059, -465.0297241, 1861.8288574, -2138.3325195, 1588.4357910
1: -445.6003113, 1243.0693359, -751.2864380, 2067.7854004, -2513.3857422, 1994.3557129
2: -332.4741821, 1432.1223145, -561.8224487, 2376.0861816, -2708.5603027, 1993.9448242
3: -715.2846069, 1279.2182617, -1210.0428467, 2132.8491211, -2848.1335449, 2489.2612305
4: -572.5853882, 1339.6010742, -967.2998047, 2226.8784180, -2799.4636230, 2306.9006348

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B2_A1_B2_B1

### Relational analysis result of IS_A1_B1_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7255619, upper bound: 1781.7129796
time: 0.67 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7252580, upper bound: 1781.7118247
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7263582, upper bound: 1781.7146671
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -371.7669678, 1491.1092529, -464.6970520, 1860.5928955, -2232.3596191, 1955.8061523
1: -601.8331909, 1655.6805420, -750.7463379, 2066.2927246, -2668.1259766, 2406.4267578
2: -449.8236389, 1903.4307861, -561.3922119, 2374.5144043, -2824.3381348, 2464.8227539
3: -969.5072021, 1708.3220215, -1209.1827393, 2131.3957520, -3100.4687500, 2917.2922363
4: -774.0858765, 1784.8104248, -966.5844116, 2225.3842773, -2999.4699707, 2751.3947754

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7099713, upper bound: 1781.7169871
time: 0.61 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7075726, upper bound: 1781.7168629
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7030888, upper bound: 1781.7043955
time: 0.55 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7030888, upper bound: 1781.7057343
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -371.5603027, 1490.2799072, -465.0297241, 1861.8288574, -2233.3884277, 1955.3095703
1: -601.4991455, 1654.7663574, -751.2864380, 2067.7854004, -2669.2846680, 2406.0527344
2: -449.5750427, 1902.3724365, -561.8224487, 2376.0861816, -2825.6611328, 2464.1943359
3: -968.9721680, 1707.3695068, -1210.0428467, 2132.8491211, -3101.5559082, 2917.2766113
4: -773.6554565, 1783.8138428, -967.2998047, 2226.8784180, -3000.5339355, 2751.1137695

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7030888, upper bound: 1781.7043955
time: 0.55 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7030888, upper bound: 1781.7057343
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -276.6474304, 1123.9506836, -331.0666504, 1336.9929199, -1613.6403809, 1455.0173340
1: -445.8388367, 1243.6837158, -533.9237671, 1482.3168945, -1928.1557617, 1777.6072998
2: -332.6482849, 1432.8239746, -398.7275696, 1705.2055664, -2037.8538818, 1831.5515137
3: -715.6668701, 1279.8508301, -860.2004395, 1524.3692627, -2240.0354004, 2140.0512695
4: -572.8852539, 1340.2568359, -686.3527222, 1595.9561768, -2168.8413086, 2026.6094971

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7150718, upper bound: 1781.7268220
time: 0.75 seconds

## Relational analysis of IS_A1_B2_B1_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_B1_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7147949, upper bound: 1781.7249399
time: 0.72 seconds

## Relational analysis of IS_A1_B2_B1_B1_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7154365, upper bound: 1781.7252314
time: 0.52 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 6.21 seconds
IS_A1_B1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7340729, upper bound: 1781.7349264
IS_A1_B1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7351206, upper bound: 1781.7349489
IS_A1_B1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7340729, upper bound: 1781.7350328
IS_A1_B1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7351206, upper bound: 1781.7350553
IS_A1_B1_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7347568, upper bound: 1781.7322319
IS_A1_B1_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7357313, upper bound: 1781.7352913
IS_A1_B1_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7333896, upper bound: 1781.7326035
IS_A1_B1_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7333798, upper bound: 1781.7332625
IS_A1_B1_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7112971, upper bound: 1781.7224045
IS_A1_B1_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7109973, upper bound: 1781.7223404
IS_A1_B1_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7136280, upper bound: 1781.7250829
IS_A1_B1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7151107, upper bound: 1781.7254262
IS_A1_B1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7069996, upper bound: 1781.7139948
IS_A1_B1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7069996, upper bound: 1781.7139948
IS_A1_B1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7095429, upper bound: 1781.7153336
IS_A1_B1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7095429, upper bound: 1781.7153336
IS_A1_B1_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7265630, upper bound: 1781.7261271
IS_A1_B1_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7265630, upper bound: 1781.7264829
IS_A1_B1_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7307373, upper bound: 1781.7265313
IS_A1_B1_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7307373, upper bound: 1781.7268866
IS_A1_B1_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7067194, upper bound: 1781.7174102
IS_A1_B1_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7067194, upper bound: 1781.7174102
IS_A1_B1_B2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7107829, upper bound: 1781.7178425
IS_A1_B1_B2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7106970, upper bound: 1781.7178425
IS_A1_B1_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7281055, upper bound: 1781.7216503
IS_A1_B1_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7282788, upper bound: 1781.7232797
IS_A1_B1_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7252580, upper bound: 1781.7118247
IS_A1_B1_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7263582, upper bound: 1781.7146671
IS_A1_B1_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7030888, upper bound: 1781.7043955
IS_A1_B1_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7030888, upper bound: 1781.7057343
IS_A1_B1_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7030888, upper bound: 1781.7043955
IS_A1_B1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7030888, upper bound: 1781.7057343
IS_A1_B2_B1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7147949, upper bound: 1781.7249399
IS_A1_B2_B1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 0, lower bound: -1781.7154365, upper bound: 1781.7252314
IS_A1_B2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7160039, upper bound: 1781.7272486
IS_A1_B2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7161104, upper bound: 1781.7270115
IS_A1_B2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7161104, upper bound: 1781.7270115
IS_A1_B2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7160039, upper bound: 1781.7272486
IS_A1_B2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7160039, upper bound: 1781.7272486
IS_A1_B2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7161104, upper bound: 1781.7268988
IS_A1_B2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7161104, upper bound: 1781.7268988
IS_A1_B2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7044850, upper bound: 1781.7032804
IS_A1_B2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7058238, upper bound: 1781.7058238
IS_A1_B2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7044850, upper bound: 1781.7032804
IS_A1_B2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7058238, upper bound: 1781.7058238
IS_A1_B2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7054382, upper bound: 1781.7053345
IS_A1_B2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7056665, upper bound: 1781.7058196
IS_A1_B2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7054382, upper bound: 1781.7053345
IS_A1_B2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7056665, upper bound: 1781.7058196
IS_A2_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7269907, upper bound: 1781.7159641
IS_A2_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7266019, upper bound: 1781.7160553
IS_A2_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7269907, upper bound: 1781.7159641
IS_A2_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7266019, upper bound: 1781.7160553
IS_A2_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7178425, upper bound: 1781.7107829
IS_A2_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7178425, upper bound: 1781.7107829
IS_A2_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7180200, upper bound: 1781.7110502
IS_A2_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7180200, upper bound: 1781.7110502
IS_A2_B1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7272483, upper bound: 1781.7160039
IS_A2_B1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7272483, upper bound: 1781.7160039
IS_A2_B1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7268988, upper bound: 1781.7161104
IS_A2_B1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7268988, upper bound: 1781.7161104
IS_A2_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7053345, upper bound: 1781.7054382
IS_A2_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7058196, upper bound: 1781.7056665
IS_A2_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7053345, upper bound: 1781.7054382
IS_A2_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7058196, upper bound: 1781.7056665
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7232127, upper bound: 1781.7228391
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7232127, upper bound: 1781.7228391
IS_A2_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7218264, upper bound: 1781.7217367
IS_A2_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7220039, upper bound: 1781.7219876
IS_A2_B2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7269576, upper bound: 1781.7313763
IS_A2_B2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7270641, upper bound: 1781.7308827
IS_A2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7191779, upper bound: 1781.7110502
IS_A2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7191779, upper bound: 1781.7110502
IS_A2_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7263118, upper bound: 1781.7152136
IS_A2_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7258625, upper bound: 1781.7153161
IS_A2_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7283244, upper bound: 1781.7159918
IS_A2_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7283244, upper bound: 1781.7159918
IS_A2_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7059145, upper bound: 1781.7059145
IS_A2_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7059145, upper bound: 1781.7059145
IS_A2_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7059145, upper bound: 1781.7070154
IS_A2_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -1781.7059145, upper bound: 1781.7070154
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=2073.053466796875
rel_dist={0: [-1781.7419244201583, 1781.7419244201583]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7403847, upper bound: 1781.7403847
time: 0.57 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7403847, upper bound: 1781.7403847
time: 0.58 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.28 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.28
Output dim: 0, lower bound: -1781.7403847, upper bound: 1781.7403847
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.28
Output dim: 0, lower bound: -1781.7403847, upper bound: 1781.7403847

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -292.2078247, 1185.7346191, -401.8600769, 1625.5317383, -1917.7395020, 1587.5942383
1: -470.8653564, 1312.1569824, -646.5264893, 1799.2055664, -2270.0708008, 1958.6832275
2: -351.4121094, 1511.5142822, -482.8912659, 2071.7290039, -2423.1408691, 1994.4055176
3: -755.9409180, 1351.1541748, -1038.5461426, 1854.7730713, -2610.7138672, 2389.6999512
4: -605.2770386, 1414.5281982, -832.4756470, 1939.4766846, -2544.7536621, 2247.0039062

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7393295, upper bound: 1781.7391794
time: 0.83 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7394478, upper bound: 1781.7394478
time: 0.55 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -407.7161560, 1648.8684082, -409.8542175, 1658.0067139, -2065.7229004, 2058.7224121
1: -656.0112915, 1825.3117676, -659.4365234, 1835.1711426, -2491.1821289, 2484.7482910
2: -490.0357056, 2101.4226074, -492.5950928, 2113.0966797, -2603.1323242, 2594.0175781
3: -1053.7666016, 1881.9282227, -1059.1666260, 1891.9827881, -2945.7492676, 2941.0947266
4: -844.8474731, 1967.4259033, -849.2470093, 1978.1652832, -2823.0124512, 2816.6723633

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7403847, upper bound: 1781.7403847
time: 0.59 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7403847, upper bound: 1781.7403847
time: 0.57 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.45 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.45
Output dim: 0, lower bound: -1781.7393295, upper bound: 1781.7391794
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.45
Output dim: 0, lower bound: -1781.7394478, upper bound: 1781.7394478
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.45
Output dim: 0, lower bound: -1781.7403847, upper bound: 1781.7403847
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.45
Output dim: 0, lower bound: -1781.7403847, upper bound: 1781.7403847

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -285.8988953, 1160.2062988, -374.1450806, 1512.7781982, -1798.6771240, 1534.3513184
1: -460.6930542, 1283.9232178, -602.2437744, 1674.7860107, -2135.4790039, 1886.1668701
2: -343.7970276, 1479.0144043, -449.8688354, 1928.4483643, -2272.2453613, 1928.8833008
3: -739.5973511, 1322.0988770, -967.2557373, 1726.6652832, -2466.2622070, 2289.3544922
4: -592.1936646, 1384.1098633, -775.8203125, 1805.0687256, -2397.2619629, 2159.9299316

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7366398, upper bound: 1781.7360097
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7366495, upper bound: 1781.7361885
time: 0.60 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -290.5298462, 1178.6455078, -427.8232422, 1729.8629150, -2020.3927002, 1606.4687500
1: -468.1905212, 1304.4074707, -687.5839844, 1915.6551514, -2383.8454590, 1991.9914551
2: -349.4006348, 1502.5856934, -513.7957153, 2205.1147461, -2554.5153809, 2016.3812256
3: -751.7043457, 1342.9464111, -1105.8681641, 1973.0588379, -2724.7631836, 2448.8144531
4: -601.8133545, 1406.0264893, -886.3011475, 2063.0727539, -2664.8859863, 2292.3276367

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7128518, upper bound: 1781.7211695
time: 0.73 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7068830, upper bound: 1781.7069113
time: 0.58 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -407.7161560, 1648.8684082, -292.2078247, 1185.7346191, -1593.4508057, 1941.0761719
1: -656.0112915, 1825.3117676, -470.8653564, 1312.1569824, -1968.1682129, 2296.1772461
2: -490.0357056, 2101.4226074, -351.4121094, 1511.5142822, -2001.5499268, 2452.8347168
3: -1053.7666016, 1881.9282227, -755.9409180, 1351.1541748, -2404.9206543, 2637.8691406
4: -844.8474731, 1967.4259033, -605.2770386, 1414.5281982, -2259.3757324, 2572.7026367

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7391794, upper bound: 1781.7393295
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7394478, upper bound: 1781.7394478
time: 0.64 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -407.7161560, 1648.8684082, -407.7161560, 1648.8684082, -2056.5844727, 2056.5844727
1: -656.0112915, 1825.3117676, -656.0112915, 1825.3117676, -2481.3229980, 2481.3229980
2: -490.0357056, 2101.4226074, -490.0357056, 2101.4226074, -2591.4582520, 2591.4582520
3: -1053.7666016, 1881.9282227, -1053.7666016, 1881.9282227, -2935.6948242, 2935.6948242
4: -844.8474731, 1967.4259033, -844.8474731, 1967.4259033, -2812.2729492, 2812.2729492

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7391794, upper bound: 1781.7393295
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7394478, upper bound: 1781.7394478
time: 0.59 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.45 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 0, lower bound: -1781.7366398, upper bound: 1781.7360097
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 0, lower bound: -1781.7366495, upper bound: 1781.7361885
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 0, lower bound: -1781.7128518, upper bound: 1781.7211695
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 0, lower bound: -1781.7068830, upper bound: 1781.7069113
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 0, lower bound: -1781.7391794, upper bound: 1781.7393295
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 0, lower bound: -1781.7394478, upper bound: 1781.7394478
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 0, lower bound: -1781.7391794, upper bound: 1781.7393295
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 0, lower bound: -1781.7394478, upper bound: 1781.7394478

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -271.1292725, 1100.4921875, -368.7804871, 1491.0291748, -1762.1584473, 1469.2724609
1: -437.1828003, 1217.7006836, -593.6539917, 1650.8096924, -2087.9924316, 1811.3546143
2: -326.1268921, 1402.7838135, -443.4127502, 1900.7370605, -2226.8637695, 1846.1965332
3: -702.1505737, 1253.0335693, -953.7003784, 1701.5428467, -2403.6933594, 2206.7338867
4: -561.1473389, 1312.8623047, -764.5543823, 1779.0599365, -2340.2070312, 2077.4167480

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7296598, upper bound: 1781.7239814
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7363330, upper bound: 1781.7360097
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7363330, upper bound: 1781.7360097
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -288.3434753, 1166.3331299, -371.0702209, 1500.1893311, -1788.5328369, 1537.4033203
1: -465.4247131, 1290.8485107, -597.2415771, 1660.8804932, -2126.3046875, 1888.0898438
2: -347.0434875, 1487.2238770, -446.1492004, 1912.4168701, -2259.4604492, 1933.3730469
3: -746.8818359, 1329.8518066, -959.3854980, 1712.2470703, -2459.1286621, 2289.2373047
4: -596.6817017, 1393.0416260, -769.4824219, 1790.0412598, -2386.7224121, 2162.5236816

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7365810, upper bound: 1781.7361885
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7365810, upper bound: 1781.7361885
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -289.4158936, 1174.1878662, -413.2554932, 1670.9244385, -1960.3402100, 1587.4433594
1: -466.3940735, 1299.4840088, -664.1547852, 1850.6903076, -2317.0844727, 1963.6387939
2: -348.0554199, 1496.9173584, -496.2336731, 2129.9567871, -2478.0119629, 1993.1510010
3: -748.8175049, 1337.8349609, -1068.2823486, 1905.7623291, -2654.5795898, 2406.1171875
4: -599.4929810, 1400.6757812, -856.0178223, 1992.6663818, -2592.1594238, 2256.6928711

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7128518, upper bound: 1781.7211695
time: 0.77 seconds

## Relational analysis of IS_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7128518, upper bound: 1781.7211695
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -281.6497803, 1142.5954590, -481.7319031, 1933.6131592, -2215.2624512, 1624.3273926
1: -453.7543945, 1264.9844971, -777.0700684, 2146.1752930, -2599.9296875, 2042.0545654
2: -338.6779480, 1456.7368164, -581.4313354, 2466.7836914, -2805.4616699, 2038.1682129
3: -728.8583374, 1301.8685303, -1251.3416748, 2214.8833008, -2943.7416992, 2553.2094727
4: -583.4093018, 1362.6614990, -1001.9818726, 2311.8371582, -2895.2460938, 2364.6433105

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7068830, upper bound: 1781.7069113
time: 0.57 seconds

## Relational analysis of IS_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7068830, upper bound: 1781.7069113
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -380.0692139, 1536.6132812, -285.8988953, 1160.2062988, -1540.2752686, 1822.5122070
1: -611.8825684, 1701.3682861, -460.6930542, 1283.9232178, -1895.8057861, 2162.0607910
2: -457.1301575, 1958.7752686, -343.7970276, 1479.0144043, -1936.1442871, 2302.5722656
3: -982.6779175, 1754.2751465, -739.5973511, 1322.0988770, -2304.7763672, 2493.8718262
4: -788.3563232, 1833.5460205, -592.1936646, 1384.1098633, -2172.4658203, 2425.7397461

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7208781, upper bound: 1781.7127374
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7158807, upper bound: 1781.7100236
time: 0.48 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -433.4168091, 1752.8944092, -290.5298462, 1178.6455078, -1612.0622559, 2043.4241943
1: -696.7171021, 1941.2100830, -468.1905212, 1304.4074707, -2001.1245117, 2409.4006348
2: -520.5569458, 2234.4389648, -349.4006348, 1502.5856934, -2023.1425781, 2583.8395996
3: -1120.6684570, 1999.0828857, -751.7043457, 1342.9464111, -2463.6147461, 2750.7871094
4: -897.9285889, 2090.1264648, -601.8133545, 1406.0264893, -2303.9550781, 2691.9396973

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7211695, upper bound: 1781.7128518
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7069113, upper bound: 1781.7068830
time: 0.51 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -380.0692139, 1536.6132812, -399.2962341, 1614.5493164, -1994.6185303, 1935.9095459
1: -611.8825684, 1701.3682861, -642.5704346, 1787.4582520, -2399.3405762, 2343.9382324
2: -457.1301575, 1958.7752686, -479.9982910, 2057.8203125, -2514.9504395, 2438.7734375
3: -982.6779175, 1754.2751465, -1032.1456299, 1842.9288330, -2825.6066895, 2786.4199219
4: -788.3563232, 1833.5460205, -827.6176147, 1926.5274658, -2714.8835449, 2661.1635742

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7390354, upper bound: 1781.7390604
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7390354, upper bound: 1781.7393295
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -433.4168091, 1752.8944092, -406.3652954, 1643.4683838, -2076.8850098, 2159.2597656
1: -696.7171021, 1941.2100830, -653.8065796, 1819.3048096, -2516.0219727, 2595.0166016
2: -520.5569458, 2234.4389648, -488.3619995, 2094.4963379, -2615.0529785, 2722.8010254
3: -1120.6684570, 1999.0828857, -1050.2325439, 1875.6983643, -2996.3666992, 3049.3154297
4: -897.9285889, 2090.1264648, -841.9771118, 1960.8830566, -2858.8115234, 2932.1035156

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7390354, upper bound: 1781.7390620
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7390354, upper bound: 1781.7394478
time: 0.83 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.81 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -1781.7363330, upper bound: 1781.7360097
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -1781.7363330, upper bound: 1781.7360097
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -1781.7365810, upper bound: 1781.7361885
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -1781.7365810, upper bound: 1781.7361885
IS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -1781.7128518, upper bound: 1781.7211695
IS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -1781.7128518, upper bound: 1781.7211695
IS_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -1781.7068830, upper bound: 1781.7069113
IS_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -1781.7068830, upper bound: 1781.7069113
IS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -1781.7208781, upper bound: 1781.7127374
IS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -1781.7158807, upper bound: 1781.7100236
IS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -1781.7211695, upper bound: 1781.7128518
IS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -1781.7069113, upper bound: 1781.7068830
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -1781.7390354, upper bound: 1781.7390604
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -1781.7390354, upper bound: 1781.7393295
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -1781.7390354, upper bound: 1781.7390620
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.81
Output dim: 0, lower bound: -1781.7390354, upper bound: 1781.7394478

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -271.1292725, 1100.4921875, -358.9670410, 1451.1046143, -1722.2338867, 1459.4592285
1: -437.1828003, 1217.7006836, -577.9741821, 1606.8073730, -2043.9902344, 1795.6748047
2: -326.1268921, 1402.7838135, -431.6136169, 1849.8392334, -2175.9660645, 1834.3974609
3: -702.1505737, 1253.0335693, -928.9398804, 1655.4935303, -2357.6440430, 2181.9733887
4: -561.1473389, 1312.8623047, -743.9436035, 1731.3353271, -2292.4821777, 2056.8059082

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7347126, upper bound: 1781.7337226
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7346046, upper bound: 1781.7344362
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -271.1292725, 1100.4921875, -371.6881714, 1501.0717773, -1772.2009277, 1472.1799316
1: -437.1828003, 1217.7006836, -598.9918213, 1662.3537598, -2099.5363770, 1816.6922607
2: -326.1268921, 1402.7838135, -447.1352234, 1913.5861816, -2239.7131348, 1849.9189453
3: -702.1505737, 1253.0335693, -962.0864258, 1714.2535400, -2416.4038086, 2215.1198730
4: -561.1473389, 1312.8623047, -770.4957886, 1792.3231201, -2353.4697266, 2083.3581543

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7347126, upper bound: 1781.7343337
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7346046, upper bound: 1781.7350503
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -288.3434753, 1166.3331299, -358.9670410, 1451.1046143, -1739.4481201, 1525.3001709
1: -465.4247131, 1290.8485107, -577.9741821, 1606.8073730, -2072.2319336, 1868.8227539
2: -347.0434875, 1487.2238770, -431.6136169, 1849.8392334, -2196.8828125, 1918.8375244
3: -746.8818359, 1329.8518066, -928.9398804, 1655.4935303, -2402.3754883, 2258.7917480
4: -596.6817017, 1393.0416260, -743.9436035, 1731.3353271, -2328.0166016, 2136.9851074

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7364392, upper bound: 1781.7355076
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7365810, upper bound: 1781.7361885
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -288.3434753, 1166.3331299, -371.6881714, 1501.0717773, -1789.4152832, 1538.0211182
1: -465.4247131, 1290.8485107, -598.9918213, 1662.3537598, -2127.7783203, 1889.8402100
2: -347.0434875, 1487.2238770, -447.1352234, 1913.5861816, -2260.6296387, 1934.3590088
3: -746.8818359, 1329.8518066, -962.0864258, 1714.2535400, -2461.1352539, 2291.9382324
4: -596.6817017, 1393.0416260, -770.4957886, 1792.3231201, -2389.0041504, 2163.5373535

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7364392, upper bound: 1781.7355336
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7365810, upper bound: 1781.7361885
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -289.4158936, 1174.1878662, -345.9256897, 1397.2662354, -1686.6821289, 1520.1134033
1: -466.3940735, 1299.4840088, -557.6153564, 1548.8400879, -2015.2341309, 1857.0993652
2: -348.0554199, 1496.9173584, -416.4815063, 1781.9025879, -2129.9575195, 1913.3989258
3: -748.8175049, 1337.8349609, -897.8513184, 1593.7917480, -2342.6088867, 2235.6862793
4: -599.4929810, 1400.6757812, -717.3684082, 1667.8487549, -2267.3417969, 2118.0441895

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B1_B1_A1

### Relational analysis result of IS_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7128518, upper bound: 1781.7211695
time: 0.52 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2

### Relational analysis result of IS_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7128518, upper bound: 1781.7211695
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -289.4158936, 1174.1878662, -418.1317444, 1690.9853516, -1980.4011230, 1592.3195801
1: -466.3940735, 1299.4840088, -672.1157227, 1873.0083008, -2339.4023438, 1971.5997314
2: -348.0554199, 1496.9173584, -502.1312866, 2155.4829102, -2503.5380859, 1999.0484619
3: -748.8175049, 1337.8349609, -1081.2238770, 1928.4916992, -2677.3090820, 2419.0588379
4: -599.4929810, 1400.6757812, -866.1642456, 2016.1826172, -2615.6755371, 2266.8395996

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B1_B2_A1

### Relational analysis result of IS_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7128518, upper bound: 1781.7211695
time: 0.52 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2

### Relational analysis result of IS_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7128518, upper bound: 1781.7211695
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -281.6497803, 1142.5954590, -396.1172180, 1589.1738281, -1870.8236084, 1538.7126465
1: -453.7543945, 1264.9844971, -639.6358643, 1764.6832275, -2218.4375000, 1904.6203613
2: -338.6779480, 1456.7368164, -478.4788208, 2027.5914307, -2366.2692871, 1935.2154541
3: -728.8583374, 1301.8685303, -1030.9722900, 1821.7375488, -2550.5959473, 2332.8405762
4: -583.4093018, 1362.6614990, -824.2117920, 1902.0307617, -2485.4394531, 2186.8732910

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7068830, upper bound: 1781.7069113
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2

### Relational analysis result of IS_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7068830, upper bound: 1781.7069113
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -281.6497803, 1142.5954590, -483.0009155, 1938.7708740, -2220.4201660, 1625.5964355
1: -453.7543945, 1264.9844971, -779.0424194, 2151.9399414, -2605.6943359, 2044.0268555
2: -338.6779480, 1456.7368164, -582.8635864, 2473.3574219, -2812.0351562, 2039.6003418
3: -728.8583374, 1301.8685303, -1254.5689697, 2220.6840820, -2949.5424805, 2556.4375000
4: -583.4093018, 1362.6614990, -1004.4208984, 2317.7651367, -2901.1743164, 2367.0825195

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7068830, upper bound: 1781.7069113
time: 0.58 seconds

## Relational analysis of IS_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7068830, upper bound: 1781.7069113
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -366.0052185, 1479.9211426, -284.7190552, 1155.4912109, -1521.4960938, 1764.6400146
1: -589.1425781, 1638.8649902, -458.7923584, 1278.7148438, -1867.8574219, 2097.6572266
2: -440.0267944, 1886.3773193, -342.3705139, 1473.0172119, -1913.0439453, 2228.7478027
3: -946.3379517, 1689.0131836, -736.5424194, 1316.6755371, -2263.0134277, 2425.5556641
4: -758.7982178, 1765.4758301, -589.7308960, 1378.4438477, -2137.2419434, 2355.2067871

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7158807, upper bound: 1781.7100236
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2

### Relational analysis result of IS_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7158807, upper bound: 1781.7100236
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -460.4322510, 1844.2117920, -277.5374146, 1126.2556152, -1586.6878662, 2121.7492676
1: -744.2117920, 2047.9725342, -447.1066895, 1246.7873535, -1990.9991455, 2495.0791016
2: -556.5748291, 2353.6630859, -333.7071533, 1435.8405762, -1992.4154053, 2687.3701172
3: -1198.3090820, 2113.7905273, -718.1170044, 1283.3281250, -2481.3740234, 2831.9072266
4: -958.5090332, 2206.7712402, -574.8683472, 1343.2436523, -2301.7521973, 2781.6391602

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7158807, upper bound: 1781.7100236
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7158807, upper bound: 1781.7100236
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -418.1317444, 1690.9853516, -289.4158936, 1174.1878662, -1592.3195801, 1980.4011230
1: -672.1157227, 1873.0083008, -466.3940735, 1299.4840088, -1971.5997314, 2339.4023438
2: -502.1312866, 2155.4829102, -348.0554199, 1496.9173584, -1999.0484619, 2503.5380859
3: -1081.2238770, 1928.4916992, -748.8175049, 1337.8349609, -2419.0588379, 2677.3090820
4: -866.1642456, 2016.1826172, -599.4929810, 1400.6757812, -2266.8398438, 2615.6755371

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7069113, upper bound: 1781.7068830
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7069113, upper bound: 1781.7068830
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -483.0009155, 1938.7708740, -281.6497803, 1142.5954590, -1625.5964355, 2220.4204102
1: -779.0424194, 2151.9399414, -453.7543945, 1264.9844971, -2044.0268555, 2605.6943359
2: -582.8635864, 2473.3574219, -338.6779480, 1456.7368164, -2039.6003418, 2812.0351562
3: -1254.5689697, 2220.6840820, -728.8583374, 1301.8685303, -2556.4375000, 2949.5424805
4: -1004.4208984, 2317.7651367, -583.4093018, 1362.6614990, -2367.0825195, 2901.1743164

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7069113, upper bound: 1781.7068830
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7069113, upper bound: 1781.7068830
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -380.0692139, 1536.6132812, -380.0692139, 1536.6132812, -1916.6823730, 1916.6823730
1: -611.8825684, 1701.3682861, -611.8825684, 1701.3682861, -2313.2507324, 2313.2507324
2: -457.1301575, 1958.7752686, -457.1301575, 1958.7752686, -2415.9055176, 2415.9055176
3: -982.6779175, 1754.2751465, -982.6779175, 1754.2751465, -2736.9521484, 2736.9523926
4: -788.3563232, 1833.5460205, -788.3563232, 1833.5460205, -2621.9020996, 2621.9020996

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7209882, upper bound: 1781.7123129
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7158448, upper bound: 1781.7095076
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -380.0692139, 1536.6132812, -433.4168091, 1752.8944092, -2132.9636230, 1970.0300293
1: -611.8825684, 1701.3682861, -696.7171021, 1941.2100830, -2553.0927734, 2398.0849609
2: -457.1301575, 1958.7752686, -520.5569458, 2234.4389648, -2691.5690918, 2479.3320312
3: -982.6779175, 1754.2751465, -1120.6684570, 1999.0828857, -2981.7607422, 2874.9433594
4: -788.3563232, 1833.5460205, -897.9285889, 2090.1264648, -2878.4824219, 2731.4746094

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7246206, upper bound: 1781.7314926
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7158448, upper bound: 1781.7100236
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -433.4168091, 1752.8944092, -380.0692139, 1536.6132812, -1970.0300293, 2132.9636230
1: -696.7171021, 1941.2100830, -611.8825684, 1701.3682861, -2398.0852051, 2553.0927734
2: -520.5569458, 2234.4389648, -457.1301575, 1958.7752686, -2479.3320312, 2691.5690918
3: -1120.6684570, 1999.0828857, -982.6779175, 1754.2751465, -2874.9436035, 2981.7607422
4: -897.9285889, 2090.1264648, -788.3563232, 1833.5460205, -2731.4746094, 2878.4824219

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7208827, upper bound: 1781.7122783
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7058357, upper bound: 1781.7058357
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -433.4168091, 1752.8944092, -433.4168091, 1752.8944092, -2186.3112793, 2186.3112793
1: -696.7171021, 1941.2100830, -696.7171021, 1941.2100830, -2637.9272461, 2637.9272461
2: -520.5569458, 2234.4389648, -520.5569458, 2234.4389648, -2754.9956055, 2754.9956055
3: -1120.6684570, 1999.0828857, -1120.6684570, 1999.0828857, -3119.7514648, 3119.7514648
4: -897.9285889, 2090.1264648, -897.9285889, 2090.1264648, -2988.0549316, 2988.0549316

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7208827, upper bound: 1781.7128235
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7058357, upper bound: 1781.7068830
time: 0.56 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.75 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 0, lower bound: -1781.7347126, upper bound: 1781.7337226
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 0, lower bound: -1781.7346046, upper bound: 1781.7344362
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 0, lower bound: -1781.7347126, upper bound: 1781.7343337
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 0, lower bound: -1781.7346046, upper bound: 1781.7350503
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 0, lower bound: -1781.7364392, upper bound: 1781.7355076
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 0, lower bound: -1781.7365810, upper bound: 1781.7361885
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 0, lower bound: -1781.7364392, upper bound: 1781.7355336
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 0, lower bound: -1781.7365810, upper bound: 1781.7361885
IS_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 0, lower bound: -1781.7128518, upper bound: 1781.7211695
IS_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 0, lower bound: -1781.7128518, upper bound: 1781.7211695
IS_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 0, lower bound: -1781.7128518, upper bound: 1781.7211695
IS_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 0, lower bound: -1781.7128518, upper bound: 1781.7211695
IS_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 0, lower bound: -1781.7068830, upper bound: 1781.7069113
IS_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 0, lower bound: -1781.7068830, upper bound: 1781.7069113
IS_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 0, lower bound: -1781.7068830, upper bound: 1781.7069113
IS_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 0, lower bound: -1781.7068830, upper bound: 1781.7069113
IS_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 0, lower bound: -1781.7158807, upper bound: 1781.7100236
IS_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 0, lower bound: -1781.7158807, upper bound: 1781.7100236
IS_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 0, lower bound: -1781.7158807, upper bound: 1781.7100236
IS_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 0, lower bound: -1781.7158807, upper bound: 1781.7100236
IS_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 0, lower bound: -1781.7069113, upper bound: 1781.7068830
IS_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 0, lower bound: -1781.7069113, upper bound: 1781.7068830
IS_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 0, lower bound: -1781.7069113, upper bound: 1781.7068830
IS_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 0, lower bound: -1781.7069113, upper bound: 1781.7068830
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 0, lower bound: -1781.7209882, upper bound: 1781.7123129
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 0, lower bound: -1781.7158448, upper bound: 1781.7095076
IS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 0, lower bound: -1781.7246206, upper bound: 1781.7314926
IS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 0, lower bound: -1781.7158448, upper bound: 1781.7100236
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 0, lower bound: -1781.7208827, upper bound: 1781.7122783
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 0, lower bound: -1781.7058357, upper bound: 1781.7058357
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 0, lower bound: -1781.7208827, upper bound: 1781.7128235
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.75
Output dim: 0, lower bound: -1781.7058357, upper bound: 1781.7068830

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -243.3785553, 992.7294312, -349.4588318, 1412.7955322, -1656.1739502, 1342.1882324
1: -393.1315308, 1095.8894043, -563.0211792, 1564.3607178, -1957.4920654, 1658.9104004
2: -292.8045654, 1266.4096680, -420.3087769, 1801.0112305, -2093.8159180, 1686.7185059
3: -630.8881836, 1126.4755859, -904.9885254, 1611.5372314, -2242.4252930, 2031.4639893
4: -503.6505737, 1183.4201660, -724.1277466, 1685.6514893, -2189.3017578, 1907.5478516

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7335601, upper bound: 1781.7333842
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7335601, upper bound: 1781.7337226
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -265.8105774, 1079.2373047, -357.5329285, 1445.2893066, -1711.0998535, 1436.7700195
1: -428.5557556, 1194.1435547, -575.6367188, 1600.3812256, -2028.9370117, 1769.7802734
2: -319.6518860, 1375.7236328, -429.8616028, 1842.4272461, -2162.0791016, 1805.5852051
3: -688.4558716, 1228.5482178, -925.2473755, 1648.8376465, -2337.2934570, 2153.7951660
4: -550.0553589, 1287.3706055, -740.9465942, 1724.3843994, -2274.4396973, 2028.3170166

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7339063, upper bound: 1781.7340704
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7339063, upper bound: 1781.7344362
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -243.3785553, 992.7294312, -361.2318726, 1458.7723389, -1702.1506348, 1353.9613037
1: -393.1315308, 1095.8894043, -582.3937378, 1615.2761230, -2008.4074707, 1678.2832031
2: -292.8045654, 1266.4096680, -434.6200256, 1859.7266846, -2152.5312500, 1701.0296631
3: -630.8881836, 1126.4755859, -935.3751221, 1665.4580078, -2296.3459473, 2061.8505859
4: -503.6505737, 1183.4201660, -748.5143433, 1741.9176025, -2245.5678711, 1931.9345703

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7336284, upper bound: 1781.7336266
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7348101, upper bound: 1781.7343337
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -265.8105774, 1079.2373047, -370.4836731, 1496.1309814, -1761.9414062, 1449.7207031
1: -428.5557556, 1194.1435547, -597.0325317, 1656.9000244, -2085.4558105, 1791.1760254
2: -319.6518860, 1375.7236328, -445.6619263, 1907.2943115, -2226.9462891, 1821.3854980
3: -688.4558716, 1228.5482178, -958.9830322, 1708.6290283, -2397.0849609, 2187.5312500
4: -550.0553589, 1287.3706055, -767.9713745, 1786.4571533, -2336.5124512, 2055.3420410

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7338005, upper bound: 1781.7344232
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7347680, upper bound: 1781.7350504
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -248.8834686, 1007.4775391, -350.0047913, 1414.9172363, -1663.8006592, 1357.4822998
1: -401.6661682, 1114.8331299, -563.3749390, 1566.8156738, -1968.4818115, 1678.2080078
2: -299.1296692, 1284.8015137, -420.7089233, 1803.8642578, -2102.9938965, 1705.5104980
3: -645.3734131, 1148.1302490, -905.8338623, 1614.2926025, -2259.6660156, 2053.9633789
4: -514.6998901, 1203.9869385, -725.4912109, 1688.1950684, -2202.8950195, 1929.4780273

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7348951, upper bound: 1781.7336982
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7356937, upper bound: 1781.7338140
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -284.9710999, 1152.5366211, -357.5548096, 1445.4300537, -1730.4008789, 1510.0914307
1: -459.8998718, 1275.6610107, -575.7282104, 1600.5986328, -2060.4985352, 1851.3891602
2: -342.9600830, 1469.4373779, -429.9432068, 1842.5012207, -2185.4609375, 1899.3806152
3: -737.6937256, 1314.7252197, -925.2229004, 1649.2312012, -2386.9248047, 2239.9482422
4: -589.6637573, 1376.7943115, -741.0708008, 1724.6430664, -2314.3063965, 2117.8652344

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7348418, upper bound: 1781.7347346
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7356284, upper bound: 1781.7346363
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -248.8834686, 1007.4775391, -362.1889648, 1462.3582764, -1711.2416992, 1369.6665039
1: -401.6661682, 1114.8331299, -583.4194946, 1619.8314209, -2021.4975586, 1698.2525635
2: -299.1296692, 1284.8015137, -435.5825806, 1864.2728271, -2163.4023438, 1720.3840332
3: -645.3734131, 1148.1302490, -937.5336304, 1670.6684570, -2316.0417480, 2085.6638184
4: -514.6998901, 1203.9869385, -751.0195923, 1746.3682861, -2261.0681152, 1955.0063477

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7356098, upper bound: 1781.7337660
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7356098, upper bound: 1781.7355336
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -284.9710999, 1152.5366211, -370.4865417, 1496.1790771, -1781.1500244, 1523.0230713
1: -459.8998718, 1275.6610107, -597.0407715, 1656.9739990, -2116.8737793, 1872.7017822
2: -342.9600830, 1469.4373779, -445.6776123, 1907.2895508, -2250.2495117, 1915.1149902
3: -737.6937256, 1314.7252197, -958.8373413, 1708.8659668, -2446.5595703, 2273.5620117
4: -589.6637573, 1376.7943115, -767.9973145, 1786.5903320, -2376.2539062, 2144.7915039

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7356254, upper bound: 1781.7338403
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7356254, upper bound: 1781.7361885
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -275.4020386, 1118.6885986, -345.9256897, 1397.2662354, -1672.6680908, 1464.6142578
1: -443.8549194, 1237.8996582, -557.6153564, 1548.8400879, -1992.6947021, 1795.5150146
2: -331.1606140, 1426.2105713, -416.4815063, 1781.9025879, -2113.0622559, 1842.6921387
3: -712.5253296, 1273.7947998, -897.8513184, 1593.7917480, -2306.3166504, 2171.6459961
4: -570.3222656, 1333.9898682, -717.3684082, 1667.8487549, -2238.1708984, 2051.3583984

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_B1_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7121027, upper bound: 1781.7206702
time: 0.54 seconds

## Relational analysis of IS_A1_B2_B1_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7118787, upper bound: 1781.7205061
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -369.6592102, 1482.5963135, -345.9256897, 1397.2662354, -1766.9254150, 1828.5218506
1: -598.4263306, 1646.3031006, -557.6153564, 1548.8400879, -2147.2663574, 2203.9184570
2: -447.2798157, 1892.5734863, -416.4815063, 1781.9025879, -2229.1816406, 2309.0549316
3: -964.0410767, 1698.5991211, -897.8513184, 1593.7917480, -2557.8322754, 2596.4504395
4: -769.7153320, 1774.5705566, -717.3684082, 1667.8487549, -2437.5639648, 2491.9389648

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B1_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7102925, upper bound: 1781.7199637
time: 0.58 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_B1_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7093235, upper bound: 1781.7193274
time: 0.57 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7118787, upper bound: 1781.7205061
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -275.4020386, 1118.6885986, -418.1317444, 1690.9853516, -1966.3870850, 1536.8203125
1: -443.8549194, 1237.8996582, -672.1157227, 1873.0083008, -2316.8632812, 1910.0153809
2: -331.1606140, 1426.2105713, -502.1312866, 2155.4829102, -2486.6433105, 1928.3416748
3: -712.5253296, 1273.7947998, -1081.2238770, 1928.4916992, -2641.0170898, 2355.0185547
4: -570.3222656, 1333.9898682, -866.1642456, 2016.1826172, -2586.5041504, 2200.1540527

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_B1_B2_A1_B1

### Relational analysis result of IS_A1_B2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7121027, upper bound: 1781.7206702
time: 0.56 seconds

## Relational analysis of IS_A1_B2_B1_B2_A1_B2

### Relational analysis result of IS_A1_B2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7104803, upper bound: 1781.7174943
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -369.6592102, 1482.5963135, -418.1317444, 1690.9853516, -2060.6445312, 1900.7280273
1: -598.4263306, 1646.3031006, -672.1157227, 1873.0083008, -2471.4345703, 2318.4187012
2: -447.2798157, 1892.5734863, -502.1312866, 2155.4829102, -2602.7624512, 2394.7048340
3: -964.0410767, 1698.5991211, -1081.2238770, 1928.4916992, -2891.2224121, 2779.8229980
4: -769.7153320, 1774.5705566, -866.1642456, 2016.1826172, -2785.8979492, 2640.7348633

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_B1_B2_A2_A1

### Relational analysis result of IS_A1_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7079371, upper bound: 1781.7162751
time: 0.59 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2_A2

### Relational analysis result of IS_A1_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7104803, upper bound: 1781.7174943
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -275.4020386, 1118.6885986, -396.1172180, 1589.1738281, -1864.5756836, 1514.8057861
1: -443.8549194, 1237.8996582, -639.6358643, 1764.6832275, -2208.5380859, 1877.5355225
2: -331.1606140, 1426.2105713, -478.4788208, 2027.5914307, -2358.7519531, 1904.6892090
3: -712.5253296, 1273.7947998, -1030.9722900, 1821.7375488, -2534.2629395, 2304.7670898
4: -570.3222656, 1333.9898682, -824.2117920, 1902.0307617, -2472.3522949, 2158.2016602

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7043871, upper bound: 1781.7030859
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7056361, upper bound: 1781.7056361
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -369.6592102, 1482.5963135, -396.1172180, 1589.1738281, -1958.8330078, 1878.7135010
1: -598.4263306, 1646.3031006, -639.6358643, 1764.6832275, -2363.1096191, 2285.9389648
2: -447.2798157, 1892.5734863, -478.4788208, 2027.5914307, -2474.8710938, 2371.0522461
3: -964.0410767, 1698.5991211, -1030.9722900, 1821.7375488, -2785.3322754, 2729.5712891
4: -769.7153320, 1774.5705566, -824.2117920, 1902.0307617, -2671.7458496, 2598.7822266

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7043871, upper bound: 1781.7030859
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7056361, upper bound: 1781.7056361
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -275.4020386, 1118.6885986, -483.0009155, 1938.7708740, -2214.1726074, 1601.6894531
1: -443.8549194, 1237.8996582, -779.0424194, 2151.9399414, -2595.7949219, 2016.9421387
2: -331.1606140, 1426.2105713, -582.8635864, 2473.3574219, -2804.5178223, 2009.0742188
3: -712.5253296, 1273.7947998, -1254.5689697, 2220.6840820, -2933.2094727, 2528.3637695
4: -570.3222656, 1333.9898682, -1004.4208984, 2317.7651367, -2888.0871582, 2338.4106445

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7052967, upper bound: 1781.7051393
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7055011, upper bound: 1781.7055973
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -369.6592102, 1482.5963135, -483.0009155, 1938.7708740, -2308.4301758, 1965.5971680
1: -598.4263306, 1646.3031006, -779.0424194, 2151.9399414, -2750.3659668, 2425.3454590
2: -447.2798157, 1892.5734863, -582.8635864, 2473.3574219, -2920.6369629, 2475.4370117
3: -964.0410767, 1698.5991211, -1254.5689697, 2220.6840820, -3182.2917480, 2953.1679688
4: -769.7153320, 1774.5705566, -1004.4208984, 2317.7651367, -3087.4804688, 2778.9914551

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7052967, upper bound: 1781.7051393
time: 0.53 seconds

## Relational analysis of IS_A1_B2_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7055011, upper bound: 1781.7055973
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -366.0052185, 1479.9211426, -270.3529968, 1098.5828857, -1464.5878906, 1750.2739258
1: -589.1425781, 1638.8649902, -435.6765442, 1215.6177979, -1804.7603760, 2074.5412598
2: -440.0267944, 1886.3773193, -325.0242615, 1400.4826660, -1840.5095215, 2211.4016113
3: -946.3379517, 1689.0131836, -699.3539429, 1250.8759766, -2197.2138672, 2388.3671875
4: -758.7982178, 1765.4758301, -559.7775879, 1309.9084473, -2068.7062988, 2325.2534180

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7193512, upper bound: 1781.7124960
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7199013, upper bound: 1781.7127374
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -366.0052185, 1479.9211426, -365.3335876, 1464.3568115, -1830.3618164, 1845.2545166
1: -589.1425781, 1638.8649902, -591.5231323, 1626.3020020, -2215.4443359, 2230.3879395
2: -440.0267944, 1886.3773193, -442.1275330, 1869.5153809, -2309.5422363, 2328.5048828
3: -946.3379517, 1689.0131836, -952.8933105, 1678.2813721, -2624.6188965, 2641.0625000
4: -758.7982178, 1765.4758301, -760.8718262, 1753.2441406, -2512.0422363, 2526.3474121

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_A1_B2_B1

### Relational analysis result of IS_A2_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7202306, upper bound: 1781.7124524
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A1_A1_B2_B1

### Relational analysis result of IS_A2_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7137218, upper bound: 1781.7067837
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2_B2

### Relational analysis result of IS_A2_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7149482, upper bound: 1781.7093298
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -460.4322510, 1844.2117920, -270.3529968, 1098.5828857, -1559.0151367, 2114.5644531
1: -744.2117920, 2047.9725342, -435.6765442, 1215.6177979, -1959.8295898, 2483.6491699
2: -556.5748291, 2353.6630859, -325.0242615, 1400.4826660, -1957.0574951, 2678.6872559
3: -1198.3090820, 2113.7905273, -699.3539429, 1250.8759766, -2449.1542969, 2813.1445312
4: -958.5090332, 2206.7712402, -559.7775879, 1309.9084473, -2268.4165039, 2766.5485840

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7149591, upper bound: 1781.7088236
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7055242, upper bound: 1781.7055172
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -460.4322510, 1844.2117920, -365.3335876, 1464.3568115, -1924.7890625, 2209.5454102
1: -744.2117920, 2047.9725342, -591.5231323, 1626.3020020, -2370.5136719, 2639.4956055
2: -556.5748291, 2353.6630859, -442.1275330, 1869.5153809, -2426.0895996, 2795.7905273
3: -1198.3090820, 2113.7905273, -952.8933105, 1678.2813721, -2875.5493164, 3064.3041992
4: -958.5090332, 2206.7712402, -760.8718262, 1753.2441406, -2711.7521973, 2967.6425781

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_A2_B2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7157784, upper bound: 1781.7100236
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_A2_B2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7147448, upper bound: 1781.7077587
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A1_A2_B2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7042834, upper bound: 1781.7029665
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7055242, upper bound: 1781.7055172
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -418.1317444, 1690.9853516, -275.4020386, 1118.6885986, -1536.8203125, 1966.3870850
1: -672.1157227, 1873.0083008, -443.8549194, 1237.8996582, -1910.0153809, 2316.8632812
2: -502.1312866, 2155.4829102, -331.1606140, 1426.2105713, -1928.3416748, 2486.6433105
3: -1081.2238770, 1928.4916992, -712.5253296, 1273.7947998, -2355.0185547, 2641.0170898
4: -866.1642456, 2016.1826172, -570.3222656, 1333.9898682, -2200.1540527, 2586.5041504

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7206702, upper bound: 1781.7121027
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7174943, upper bound: 1781.7104803
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -418.1317444, 1690.9853516, -369.6592102, 1482.5963135, -1900.7280273, 2060.6445312
1: -672.1157227, 1873.0083008, -598.4263306, 1646.3031006, -2318.4187012, 2471.4345703
2: -502.1312866, 2155.4829102, -447.2798157, 1892.5734863, -2394.7048340, 2602.7624512
3: -1081.2238770, 1928.4916992, -964.0410767, 1698.5991211, -2779.8229980, 2891.2224121
4: -866.1642456, 2016.1826172, -769.7153320, 1774.5705566, -2640.7348633, 2785.8979492

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A2_A1_B2_B1

### Relational analysis result of IS_A2_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7162751, upper bound: 1781.7079371
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_A1_B2_B2

### Relational analysis result of IS_A2_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7174943, upper bound: 1781.7104803
time: 0.50 seconds

## BFS IS instance: IS_A2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -483.0009155, 1938.7708740, -275.4020386, 1118.6885986, -1601.6894531, 2214.1728516
1: -779.0424194, 2151.9399414, -443.8549194, 1237.8996582, -2016.9421387, 2595.7949219
2: -582.8635864, 2473.3574219, -331.1606140, 1426.2105713, -2009.0742188, 2804.5178223
3: -1254.5689697, 2220.6840820, -712.5253296, 1273.7947998, -2528.3637695, 2933.2094727
4: -1004.4208984, 2317.7651367, -570.3222656, 1333.9898682, -2338.4106445, 2888.0871582

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7051393, upper bound: 1781.7052967
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7055973, upper bound: 1781.7055011
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -483.0009155, 1938.7708740, -369.6592102, 1482.5963135, -1965.5971680, 2308.4301758
1: -779.0424194, 2151.9399414, -598.4263306, 1646.3031006, -2425.3454590, 2750.3659668
2: -582.8635864, 2473.3574219, -447.2798157, 1892.5734863, -2475.4370117, 2920.6369629
3: -1254.5689697, 2220.6840820, -964.0410767, 1698.5991211, -2953.1679688, 3182.2917480
4: -1004.4208984, 2317.7651367, -769.7153320, 1774.5705566, -2778.9914551, 3087.4804688

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7051393, upper bound: 1781.7052967
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7055973, upper bound: 1781.7055011
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -366.0052185, 1479.9211426, -378.8262939, 1531.5242920, -1897.5292969, 1858.7474365
1: -589.1425781, 1638.8649902, -609.8695679, 1695.7924805, -2284.9350586, 2248.7346191
2: -440.0267944, 1886.3773193, -455.6215515, 1952.2917480, -2392.3186035, 2341.9987793
3: -946.3379517, 1689.0131836, -979.4728394, 1748.4309082, -2694.7687988, 2668.4860840
4: -758.7982178, 1765.4758301, -785.7493896, 1827.4451904, -2586.2431641, 2551.2250977

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7229079, upper bound: 1781.7226096
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7229079, upper bound: 1781.7226096
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -460.4322510, 1844.2117920, -368.4671021, 1489.3044434, -1949.7366943, 2212.6789551
1: -744.2117920, 2047.9725342, -593.0672607, 1649.6279297, -2393.8398438, 2641.0397949
2: -556.5748291, 2353.6630859, -443.2113647, 1898.5446777, -2455.1188965, 2796.8745117
3: -1198.3090820, 2113.7905273, -952.9396973, 1700.2930908, -2897.0534668, 3066.7302246
4: -958.5090332, 2206.7712402, -764.5203247, 1776.6645508, -2735.1728516, 2971.2910156

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7149640, upper bound: 1781.7149640
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7192073, upper bound: 1781.7187671
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -378.8262939, 1531.5242920, -418.1317444, 1690.9853516, -2069.8115234, 1949.6560059
1: -609.8695679, 1695.7924805, -672.1157227, 1873.0083008, -2482.8779297, 2367.9079590
2: -455.6215515, 1952.2917480, -502.1312866, 2155.4829102, -2611.1044922, 2454.4226074
3: -979.4728394, 1748.4309082, -1081.2238770, 1928.4916992, -2907.9643555, 2829.6545410
4: -785.7493896, 1827.4451904, -866.1642456, 2016.1826172, -2801.9321289, 2693.6091309

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_B1_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7242687, upper bound: 1781.7297052
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_B2

### Relational analysis result of IS_A2_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7243548, upper bound: 1781.7292878
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -368.4671021, 1489.3044434, -483.0009155, 1938.7708740, -2307.2380371, 1972.3054199
1: -593.0672607, 1649.6279297, -779.0424194, 2151.9399414, -2745.0073242, 2428.6704102
2: -443.2113647, 1898.5446777, -582.8635864, 2473.3574219, -2916.5688477, 2481.4082031
3: -952.9396973, 1700.2930908, -1254.5689697, 2220.6840820, -3173.6237793, 2954.3959961
4: -764.5203247, 1776.6645508, -1004.4208984, 2317.7651367, -3082.2854004, 2781.0854492

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7158822, upper bound: 1781.7100236
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7158822, upper bound: 1781.7100236
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -418.1317444, 1690.9853516, -378.8262939, 1531.5242920, -1949.6560059, 2069.8115234
1: -672.1157227, 1873.0083008, -609.8695679, 1695.7924805, -2367.9079590, 2482.8779297
2: -502.1312866, 2155.4829102, -455.6215515, 1952.2917480, -2454.4226074, 2611.1044922
3: -1081.2238770, 1928.4916992, -979.4728394, 1748.4309082, -2829.6545410, 2907.9643555
4: -866.1642456, 2016.1826172, -785.7493896, 1827.4451904, -2693.6091309, 2801.9321289

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7058357, upper bound: 1781.7058357
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7058357, upper bound: 1781.7058357
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -483.0009155, 1938.7708740, -368.4671021, 1489.3044434, -1972.3054199, 2307.2380371
1: -779.0424194, 2151.9399414, -593.0672607, 1649.6279297, -2428.6704102, 2745.0073242
2: -582.8635864, 2473.3574219, -443.2113647, 1898.5446777, -2481.4079590, 2916.5688477
3: -1254.5689697, 2220.6840820, -952.9396973, 1700.2930908, -2954.3959961, 3173.6237793
4: -1004.4208984, 2317.7651367, -764.5203247, 1776.6645508, -2781.0854492, 3082.2854004

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7058357, upper bound: 1781.7058357
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7058357, upper bound: 1781.7058357
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -418.1317444, 1690.9853516, -432.1498413, 1747.6823730, -2165.8139648, 2123.1352539
1: -672.1157227, 1873.0083008, -694.6755981, 1935.5135498, -2607.6289062, 2567.6838379
2: -502.1312866, 2155.4829102, -519.0313721, 2227.8112793, -2729.9421387, 2674.5141602
3: -1081.2238770, 1928.4916992, -1117.4217529, 1993.1535645, -3074.3774414, 3045.9135742
4: -866.1642456, 2016.1826172, -895.2977905, 2083.9125977, -2950.0766602, 2911.4804688

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7068830, upper bound: 1781.7068830
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7068830, upper bound: 1781.7068830
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -483.0009155, 1938.7708740, -418.9283142, 1694.6027832, -2177.6037598, 2357.6989746
1: -779.0424194, 2151.9399414, -673.4575195, 1877.1345215, -2656.1770020, 2825.3967285
2: -582.8635864, 2473.3574219, -503.1912231, 2160.2084961, -2743.0720215, 2976.5485840
3: -1254.5689697, 2220.6840820, -1083.6662598, 1932.2321777, -3185.9960938, 3304.3503418
4: -1004.4208984, 2317.7651367, -868.0754395, 2020.0151367, -3024.4357910, 3185.8403320

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7068830, upper bound: 1781.7068830
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7068830, upper bound: 1781.7068830
time: 0.54 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.61 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7335601, upper bound: 1781.7333842
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7335601, upper bound: 1781.7337226
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7339063, upper bound: 1781.7340704
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7339063, upper bound: 1781.7344362
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7336284, upper bound: 1781.7336266
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7348101, upper bound: 1781.7343337
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7338005, upper bound: 1781.7344232
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7347680, upper bound: 1781.7350504
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7348951, upper bound: 1781.7336982
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7356937, upper bound: 1781.7338140
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7348418, upper bound: 1781.7347346
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7356284, upper bound: 1781.7346363
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7356098, upper bound: 1781.7337660
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7356098, upper bound: 1781.7355336
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7356254, upper bound: 1781.7338403
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7356254, upper bound: 1781.7361885
IS_A1_B2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7121027, upper bound: 1781.7206702
IS_A1_B2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7118787, upper bound: 1781.7205061
IS_A1_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7093235, upper bound: 1781.7193274
IS_A1_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7118787, upper bound: 1781.7205061
IS_A1_B2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7121027, upper bound: 1781.7206702
IS_A1_B2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7104803, upper bound: 1781.7174943
IS_A1_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7079371, upper bound: 1781.7162751
IS_A1_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7104803, upper bound: 1781.7174943
IS_A1_B2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7043871, upper bound: 1781.7030859
IS_A1_B2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7056361, upper bound: 1781.7056361
IS_A1_B2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7043871, upper bound: 1781.7030859
IS_A1_B2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7056361, upper bound: 1781.7056361
IS_A1_B2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7052967, upper bound: 1781.7051393
IS_A1_B2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7055011, upper bound: 1781.7055973
IS_A1_B2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7052967, upper bound: 1781.7051393
IS_A1_B2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7055011, upper bound: 1781.7055973
IS_A2_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7193512, upper bound: 1781.7124960
IS_A2_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7199013, upper bound: 1781.7127374
IS_A2_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7137218, upper bound: 1781.7067837
IS_A2_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7149482, upper bound: 1781.7093298
IS_A2_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7149591, upper bound: 1781.7088236
IS_A2_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7055242, upper bound: 1781.7055172
IS_A2_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7042834, upper bound: 1781.7029665
IS_A2_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7055242, upper bound: 1781.7055172
IS_A2_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7206702, upper bound: 1781.7121027
IS_A2_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7174943, upper bound: 1781.7104803
IS_A2_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7162751, upper bound: 1781.7079371
IS_A2_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7174943, upper bound: 1781.7104803
IS_A2_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7051393, upper bound: 1781.7052967
IS_A2_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7055973, upper bound: 1781.7055011
IS_A2_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7051393, upper bound: 1781.7052967
IS_A2_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7055973, upper bound: 1781.7055011
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7229079, upper bound: 1781.7226096
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7229079, upper bound: 1781.7226096
IS_A2_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7149640, upper bound: 1781.7149640
IS_A2_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7192073, upper bound: 1781.7187671
IS_A2_B2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7242687, upper bound: 1781.7297052
IS_A2_B2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7243548, upper bound: 1781.7292878
IS_A2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7158822, upper bound: 1781.7100236
IS_A2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7158822, upper bound: 1781.7100236
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7058357, upper bound: 1781.7058357
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7058357, upper bound: 1781.7058357
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7058357, upper bound: 1781.7058357
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7058357, upper bound: 1781.7058357
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7068830, upper bound: 1781.7068830
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7068830, upper bound: 1781.7068830
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7068830, upper bound: 1781.7068830
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -1781.7068830, upper bound: 1781.7068830

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -243.3785553, 992.7294312, -332.0976868, 1347.1492920, -1590.5275879, 1324.8269043
1: -393.1315308, 1095.8894043, -535.6076660, 1489.0166016, -1882.1480713, 1631.4970703
2: -292.8045654, 1266.4096680, -399.4431458, 1718.2320557, -2011.0364990, 1665.8527832
3: -630.8881836, 1126.4755859, -859.8058472, 1532.9085693, -2163.7968750, 1986.2814941
4: -503.6505737, 1183.4201660, -688.0932617, 1606.2116699, -2109.8618164, 1871.5134277

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7313207, upper bound: 1781.7305653
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7311268, upper bound: 1781.7308018
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -243.3785553, 992.7294312, -353.7543030, 1429.9700928, -1673.3483887, 1346.4837646
1: -393.1315308, 1095.8894043, -569.5081787, 1583.4539795, -1976.5853271, 1665.3975830
2: -292.8045654, 1266.4096680, -425.2476196, 1822.9011230, -2115.7055664, 1691.6572266
3: -630.8881836, 1126.4755859, -915.5364990, 1631.3238525, -2262.2116699, 2042.0120850
4: -503.6505737, 1183.4201660, -733.0568237, 1706.1018066, -2209.7524414, 1916.4769287

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7313207, upper bound: 1781.7310247
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7311268, upper bound: 1781.7312573
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -265.8105774, 1079.2373047, -332.0976868, 1347.1492920, -1612.9598389, 1411.3347168
1: -428.5557556, 1194.1435547, -535.6076660, 1489.0166016, -1917.5723877, 1729.7512207
2: -319.6518860, 1375.7236328, -399.4431458, 1718.2320557, -2037.8837891, 1775.1667480
3: -688.4558716, 1228.5482178, -859.8058472, 1532.9085693, -2221.3645020, 2088.3540039
4: -550.0553589, 1287.3706055, -688.0932617, 1606.2116699, -2156.2666016, 1975.4638672

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7311290, upper bound: 1781.7310397
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7314278, upper bound: 1781.7316976
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -265.8105774, 1079.2373047, -353.9849243, 1430.8825684, -1696.6929932, 1433.2221680
1: -428.5557556, 1194.1435547, -569.8681030, 1584.4713135, -2013.0270996, 1764.0117188
2: -319.6518860, 1375.7236328, -425.5245056, 1824.0655518, -2143.7175293, 1801.2481689
3: -688.4558716, 1228.5482178, -916.1353149, 1632.3653564, -2320.8210449, 2144.6835938
4: -550.0553589, 1287.3706055, -733.5337524, 1707.1878662, -2257.2431641, 2020.9039307

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7311290, upper bound: 1781.7314755
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7314278, upper bound: 1781.7321684
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -235.7002716, 962.4818726, -319.7587585, 1291.1030273, -1526.8033447, 1282.2406006
1: -380.8083801, 1062.2239990, -515.3890991, 1429.5679932, -1810.3763428, 1577.6130371
2: -283.5702515, 1227.8349609, -384.3111267, 1646.2873535, -1929.8575439, 1612.1457520
3: -611.0213623, 1091.5878906, -828.7883911, 1473.8175049, -2084.8388672, 1920.3762207
4: -487.7486267, 1147.0162354, -662.3361206, 1542.5506592, -2030.2989502, 1809.3522949

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7336284, upper bound: 1781.7313963
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7312080, upper bound: 1781.7315404
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7312395, upper bound: 1781.7317296
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -241.6857300, 985.8019409, -356.4088745, 1438.8674316, -1680.5529785, 1342.2108154
1: -390.3502197, 1088.3789062, -574.4956665, 1593.3829346, -1983.7330322, 1662.8745117
2: -290.7436218, 1257.4886475, -428.7692871, 1834.1484375, -2124.8918457, 1686.2579346
3: -626.4828491, 1118.9689941, -922.3775024, 1643.5783691, -2270.0612793, 2041.3464355
4: -500.2250061, 1175.3258057, -738.4471436, 1718.5603027, -2218.7849121, 1913.7729492

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7348101, upper bound: 1781.7320057
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7324434, upper bound: 1781.7319136
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7322214, upper bound: 1781.7321241
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -257.6581726, 1046.3809814, -328.1076355, 1324.4256592, -1582.0834961, 1374.4886475
1: -415.0925598, 1157.9031982, -528.4692383, 1466.7145996, -1881.8071289, 1686.3724365
2: -309.6553650, 1333.9293213, -394.1931458, 1688.6954346, -1998.3508301, 1728.1223145
3: -667.2738647, 1191.0645752, -849.8226929, 1512.3536377, -2179.6274414, 2040.8872070
4: -533.2141113, 1248.1141357, -679.7723389, 1582.2512207, -2115.4653320, 1927.8863525

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7271173, upper bound: 1781.7204430
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7336787, upper bound: 1781.7323738
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7338005, upper bound: 1781.7340657
time: 0.51 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -264.5714111, 1074.2716064, -365.6908569, 1476.4522705, -1741.0235596, 1439.9624023
1: -426.5989380, 1188.7274170, -589.2019653, 1635.2561035, -2061.8547363, 1777.9294434
2: -318.1918335, 1369.3181152, -439.8510742, 1881.9993896, -2200.1911621, 1809.1691895
3: -685.2264404, 1223.0764160, -946.0524902, 1686.9371338, -2372.1635742, 2169.1281738
4: -547.5473633, 1281.5174561, -757.9997559, 1763.3680420, -2310.9155273, 2039.5170898

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7346688, upper bound: 1781.7329635
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7347680, upper bound: 1781.7346412
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -239.2006073, 968.1088257, -323.9312744, 1314.2329102, -1553.4334717, 1292.0400391
1: -386.3049011, 1070.9393311, -522.4434204, 1452.4851074, -1838.7900391, 1593.3826904
2: -287.5702820, 1234.7260742, -389.6179199, 1676.2990723, -1963.8692627, 1624.3439941
3: -620.6221313, 1102.7089844, -838.6573486, 1495.3729248, -2115.9948730, 1941.3663330
4: -494.2549744, 1157.1925049, -671.2069702, 1566.9478760, -2061.2026367, 1828.3994141

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7326480, upper bound: 1781.7336982
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7347909, upper bound: 1781.7332836
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7347738, upper bound: 1781.7333395
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -247.4279633, 1001.6030273, -344.9541016, 1394.3311768, -1641.7591553, 1346.5568848
1: -399.3194275, 1108.3143311, -555.1796265, 1544.0849609, -1943.4044189, 1663.4938965
2: -297.3673096, 1277.3323975, -414.5255737, 1777.6217041, -2074.9890137, 1691.8579102
3: -641.6416016, 1141.4392090, -892.8368530, 1590.7534180, -2232.3950195, 2034.2761230
4: -511.6824036, 1197.0350342, -714.9190674, 1663.6180420, -2175.3005371, 1911.9541016

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7336084, upper bound: 1781.7336873
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7354229, upper bound: 1781.7338140
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -274.7355652, 1110.5478516, -330.2367249, 1339.6118164, -1614.3474121, 1440.7845459
1: -443.4833984, 1228.7630615, -532.5385132, 1480.7984619, -1924.2817383, 1761.3015137
2: -330.6237488, 1416.0394287, -397.1693726, 1708.5131836, -2039.1369629, 1813.2087402
3: -711.3191528, 1266.3244629, -854.9274902, 1524.6968994, -2236.0161133, 2121.2517090
4: -567.9056396, 1326.9449463, -684.3079834, 1597.3702393, -2165.2758789, 2011.2529297

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7326082, upper bound: 1781.7347346
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7320633, upper bound: 1781.7314030
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7326668, upper bound: 1781.7321477
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -283.7110596, 1147.4442139, -352.5554199, 1425.1358643, -1708.8469238, 1499.9995117
1: -457.8697510, 1270.0161133, -567.5919189, 1578.1818848, -2036.0516357, 1837.6080322
2: -341.4356995, 1462.9710693, -423.8333740, 1816.6397705, -2158.0754395, 1886.8044434
3: -734.4865112, 1308.8919678, -912.3572388, 1626.0194092, -2360.5058594, 2221.2492676
4: -587.0469360, 1370.7595215, -730.6228027, 1700.4152832, -2287.4619141, 2101.3823242

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7335525, upper bound: 1781.7345300
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7353640, upper bound: 1781.7346363
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -248.8834686, 1007.4775391, -329.4342957, 1329.8442383, -1578.7276611, 1336.9118652
1: -401.6661682, 1114.8331299, -530.6300049, 1472.7001953, -1874.3662109, 1645.4628906
2: -299.1296692, 1284.8015137, -395.8134460, 1695.6004639, -1994.7299805, 1680.6149902
3: -645.3734131, 1148.1302490, -853.2512207, 1518.5123291, -2163.8857422, 2001.3812256
4: -514.6998901, 1203.9869385, -682.5424194, 1588.6904297, -2103.3903809, 1886.5292969

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7327749, upper bound: 1781.7311926
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7350633, upper bound: 1781.7332903
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 34
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 0
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 9
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 25

Time for candidate selection: 11.82 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 34

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 34

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 2

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 48

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 48

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 19

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 5

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 25

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 25

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: B, layer: 5, pos: 14
type: A, layer: 5, pos: 14
type: B, layer: 5, pos: 24
type: A, layer: 5, pos: 24
type: B, layer: 5, pos: 22
type: B, layer: 5, pos: 9
type: A, layer: 5, pos: 22
type: A, layer: 5, pos: 9
type: B, layer: 5, pos: 41
type: B, layer: 5, pos: 1
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 2
type: B, layer: 5, pos: 11
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 2
type: A, layer: 5, pos: 41
type: A, layer: 5, pos: 1
type: A, layer: 5, pos: 11
type: A, layer: 5, pos: 42
type: B, layer: 5, pos: 42
type: B, layer: 5, pos: 5
type: B, layer: 5, pos: 18
type: A, layer: 5, pos: 5
type: B, layer: 5, pos: 39
type: A, layer: 5, pos: 39
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 13
type: A, layer: 5, pos: 13
type: A, layer: 5, pos: 18
type: B, layer: 5, pos: 40
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 17
type: A, layer: 5, pos: 10
type: B, layer: 5, pos: 15
type: B, layer: 5, pos: 10
type: B, layer: 5, pos: 3
type: A, layer: 5, pos: 17
type: A, layer: 5, pos: 40
type: B, layer: 5, pos: 37
type: A, layer: 5, pos: 37
type: A, layer: 5, pos: 3
type: B, layer: 5, pos: 4
type: A, layer: 5, pos: 15
type: B, layer: 5, pos: 28
type: A, layer: 5, pos: 28
type: A, layer: 5, pos: 4
type: B, layer: 5, pos: 48
type: B, layer: 5, pos: 45
type: B, layer: 5, pos: 34
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 45
type: A, layer: 5, pos: 48
type: B, layer: 5, pos: 27
type: A, layer: 5, pos: 27
type: B, layer: 5, pos: 38
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 34
type: A, layer: 5, pos: 38
type: B, layer: 5, pos: 0
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 0
type: B, layer: 5, pos: 7
type: A, layer: 5, pos: 7
type: B, layer: 5, pos: 23
type: A, layer: 5, pos: 23
type: B, layer: 5, pos: 26
type: A, layer: 5, pos: 35
type: B, layer: 5, pos: 35
type: A, layer: 5, pos: 20
type: B, layer: 5, pos: 49
type: B, layer: 5, pos: 20
type: A, layer: 5, pos: 49
type: B, layer: 5, pos: 31
type: A, layer: 5, pos: 26
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 31
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 16
type: A, layer: 5, pos: 16
type: B, layer: 5, pos: 12
type: B, layer: 5, pos: 43
type: A, layer: 5, pos: 43
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 12
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 8
type: A, layer: 5, pos: 8
type: A, layer: 5, pos: 33
type: B, layer: 5, pos: 29
type: B, layer: 5, pos: 33
type: A, layer: 5, pos: 29
type: A, layer: 5, pos: 32
type: B, layer: 5, pos: 32
type: B, layer: 5, pos: 36
type: A, layer: 5, pos: 36
type: B, layer: 5, pos: 47
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 47

Time for candidate selection: 34.33 seconds

### Candidate
type: B, layer: 5, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 24

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 24

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 41

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 2

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 11

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 2

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 41

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 11

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 42

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 42

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 5

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 5

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 10

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 10

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 48

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 34

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 48

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 34

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 7

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 7

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 26

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 49

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 49

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 31

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 26

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 31

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 12

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 43

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 43

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 12

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 47

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 19

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 19

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 47

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 7
type: B, layer: 7, pos: 30
type: B, layer: 7, pos: 29
type: A, layer: 7, pos: 29
type: A, layer: 7, pos: 30
type: B, layer: 7, pos: 4
type: A, layer: 7, pos: 4
type: B, layer: 7, pos: 24
type: B, layer: 7, pos: 46
type: A, layer: 7, pos: 46
type: A, layer: 7, pos: 24
type: B, layer: 7, pos: 23
type: B, layer: 7, pos: 3
type: B, layer: 7, pos: 41
type: A, layer: 7, pos: 3
type: A, layer: 7, pos: 23
type: B, layer: 7, pos: 37
type: A, layer: 7, pos: 41
type: B, layer: 7, pos: 49
type: B, layer: 7, pos: 21
type: A, layer: 7, pos: 37
type: B, layer: 7, pos: 12
type: A, layer: 7, pos: 49
type: A, layer: 7, pos: 21
type: A, layer: 7, pos: 12
type: B, layer: 7, pos: 27
type: B, layer: 7, pos: 5
type: A, layer: 7, pos: 27
type: B, layer: 7, pos: 1
type: A, layer: 7, pos: 5
type: A, layer: 7, pos: 1
type: B, layer: 7, pos: 20
type: A, layer: 7, pos: 20
type: B, layer: 7, pos: 35
type: A, layer: 7, pos: 35
type: B, layer: 7, pos: 45
type: B, layer: 7, pos: 38
type: B, layer: 7, pos: 9
type: A, layer: 7, pos: 45
type: A, layer: 7, pos: 9
type: B, layer: 7, pos: 11
type: B, layer: 7, pos: 31
type: A, layer: 7, pos: 38
type: A, layer: 7, pos: 11
type: B, layer: 7, pos: 34
type: B, layer: 7, pos: 13
type: A, layer: 7, pos: 31
type: A, layer: 7, pos: 13
type: A, layer: 7, pos: 34
type: B, layer: 7, pos: 33
type: A, layer: 7, pos: 33
type: B, layer: 7, pos: 42
type: B, layer: 7, pos: 36
type: A, layer: 7, pos: 42
type: B, layer: 7, pos: 22
type: A, layer: 7, pos: 36
type: B, layer: 7, pos: 16
type: A, layer: 7, pos: 16
type: A, layer: 7, pos: 22
type: A, layer: 7, pos: 32
type: B, layer: 7, pos: 32
type: B, layer: 7, pos: 15
type: B, layer: 7, pos: 48
type: B, layer: 7, pos: 25
type: A, layer: 7, pos: 25
type: A, layer: 7, pos: 15
type: A, layer: 7, pos: 48
type: B, layer: 7, pos: 44
type: A, layer: 7, pos: 44
type: B, layer: 7, pos: 19
type: B, layer: 7, pos: 8
type: B, layer: 7, pos: 10
type: A, layer: 7, pos: 10
type: A, layer: 7, pos: 19
type: B, layer: 7, pos: 39
type: A, layer: 7, pos: 39
type: A, layer: 7, pos: 8
type: B, layer: 7, pos: 40
type: A, layer: 7, pos: 40
type: B, layer: 7, pos: 17
type: B, layer: 7, pos: 43
type: A, layer: 7, pos: 43
type: A, layer: 7, pos: 17
type: B, layer: 7, pos: 2
type: A, layer: 7, pos: 2
type: B, layer: 7, pos: 18
type: A, layer: 7, pos: 18
type: B, layer: 7, pos: 28
type: B, layer: 7, pos: 7
type: A, layer: 7, pos: 7
type: B, layer: 7, pos: 6
type: A, layer: 7, pos: 28
type: A, layer: 7, pos: 6
type: B, layer: 7, pos: 0
type: A, layer: 7, pos: 26
type: B, layer: 7, pos: 26
type: A, layer: 7, pos: 0
type: B, layer: 7, pos: 14
type: A, layer: 7, pos: 14
type: B, layer: 7, pos: 47
type: A, layer: 7, pos: 47

Time for candidate selection: 76.58 seconds

### Candidate
type: B, layer: 7, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=2073.053466796875
rel_dist={0: [-1781.7403846768325, 1781.740384676833]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7388007, upper bound: 1781.7388007
time: 0.70 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7388007, upper bound: 1781.7388007
time: 0.55 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.39 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.39
Output dim: 0, lower bound: -1781.7388007, upper bound: 1781.7388007
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.39
Output dim: 0, lower bound: -1781.7388007, upper bound: 1781.7388007

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -292.2078247, 1185.7346191, -395.0211487, 1597.6091309, -1889.8168945, 1580.7556152
1: -470.8653564, 1312.1569824, -635.5115356, 1768.3704834, -2239.2358398, 1947.6683350
2: -351.4121094, 1511.5142822, -474.6130371, 2036.1599121, -2387.5720215, 1986.1271973
3: -755.9409180, 1351.1541748, -1020.9802856, 1822.9670410, -2578.9079590, 2372.1342773
4: -605.2770386, 1414.5281982, -818.1955566, 1906.2989502, -2511.5759277, 2232.7236328

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7388007, upper bound: 1781.7388007
time: 0.64 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7388007, upper bound: 1781.7388007
time: 0.58 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -407.7161560, 1648.8684082, -409.0175171, 1654.4312744, -2062.1474609, 2057.8859863
1: -656.0112915, 1825.3117676, -658.0963745, 1831.3126221, -2487.3234863, 2483.4082031
2: -490.0357056, 2101.4226074, -491.5933533, 2108.5288086, -2598.5644531, 2593.0158691
3: -1053.7666016, 1881.9282227, -1057.0537109, 1888.0509033, -2941.8173828, 2938.9819336
4: -844.8474731, 1967.4259033, -847.5256958, 1973.9653320, -2818.8127441, 2814.9509277

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7388007, upper bound: 1781.7388007
time: 0.81 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7388007, upper bound: 1781.7388007
time: 0.72 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.90 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.90
Output dim: 0, lower bound: -1781.7388007, upper bound: 1781.7388007
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.90
Output dim: 0, lower bound: -1781.7388007, upper bound: 1781.7388007
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.90
Output dim: 0, lower bound: -1781.7388007, upper bound: 1781.7388007
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.90
Output dim: 0, lower bound: -1781.7388007, upper bound: 1781.7388007

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -292.2078247, 1185.7346191, -292.2078247, 1185.7346191, -1477.9423828, 1477.9423828
1: -470.8653564, 1312.1569824, -470.8653564, 1312.1569824, -1783.0222168, 1783.0222168
2: -351.4121094, 1511.5142822, -351.4121094, 1511.5142822, -1862.9263916, 1862.9262695
3: -755.9409180, 1351.1541748, -755.9409180, 1351.1541748, -2107.0952148, 2107.0952148
4: -605.2770386, 1414.5281982, -605.2770386, 1414.5281982, -2019.8050537, 2019.8050537

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7137639, upper bound: 1781.7180397
time: 0.50 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7107504, upper bound: 1781.7107504
time: 0.56 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -292.2078247, 1185.7346191, -407.7161560, 1648.8684082, -1941.0761719, 1593.4508057
1: -470.8653564, 1312.1569824, -656.0112915, 1825.3117676, -2296.1772461, 1968.1682129
2: -351.4121094, 1511.5142822, -490.0357056, 2101.4226074, -2452.8347168, 2001.5500488
3: -755.9409180, 1351.1541748, -1053.7666016, 1881.9282227, -2637.8691406, 2404.9206543
4: -605.2770386, 1414.5281982, -844.8474731, 1967.4259033, -2572.7026367, 2259.3757324

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7180397, upper bound: 1781.7202100
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7107504, upper bound: 1781.7153278
time: 0.57 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -407.7161560, 1648.8684082, -292.2078247, 1185.7346191, -1593.4508057, 1941.0761719
1: -656.0112915, 1825.3117676, -470.8653564, 1312.1569824, -1968.1682129, 2296.1772461
2: -490.0357056, 2101.4226074, -351.4121094, 1511.5142822, -2001.5499268, 2452.8347168
3: -1053.7666016, 1881.9282227, -755.9409180, 1351.1541748, -2404.9206543, 2637.8691406
4: -844.8474731, 1967.4259033, -605.2770386, 1414.5281982, -2259.3757324, 2572.7026367

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7377676, upper bound: 1781.7379405
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7380566, upper bound: 1781.7380566
time: 0.57 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -407.7161560, 1648.8684082, -407.7161560, 1648.8684082, -2056.5844727, 2056.5844727
1: -656.0112915, 1825.3117676, -656.0112915, 1825.3117676, -2481.3229980, 2481.3229980
2: -490.0357056, 2101.4226074, -490.0357056, 2101.4226074, -2591.4582520, 2591.4582520
3: -1053.7666016, 1881.9282227, -1053.7666016, 1881.9282227, -2935.6948242, 2935.6948242
4: -844.8474731, 1967.4259033, -844.8474731, 1967.4259033, -2812.2729492, 2812.2729492

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7377676, upper bound: 1781.7379405
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7380566, upper bound: 1781.7380566
time: 0.57 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.49 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 0, lower bound: -1781.7137639, upper bound: 1781.7180397
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 0, lower bound: -1781.7107504, upper bound: 1781.7107504
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 0, lower bound: -1781.7180397, upper bound: 1781.7202100
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 0, lower bound: -1781.7107504, upper bound: 1781.7153278
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 0, lower bound: -1781.7377676, upper bound: 1781.7379405
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 0, lower bound: -1781.7380566, upper bound: 1781.7380566
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 0, lower bound: -1781.7377676, upper bound: 1781.7379405
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 0, lower bound: -1781.7380566, upper bound: 1781.7380566

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -284.9186401, 1156.6960449, -276.7585144, 1124.4237061, -1409.3422852, 1433.4544678
1: -459.1326294, 1280.0335693, -446.0162659, 1244.2011719, -1703.3337402, 1726.0498047
2: -342.6111450, 1474.5635986, -332.7811279, 1433.4194336, -1776.0305176, 1807.3447266
3: -737.0704956, 1317.7193604, -715.9465942, 1280.3992920, -2017.4696045, 2033.6660156
4: -590.0875244, 1379.6500244, -573.1133423, 1340.8247070, -1930.9119873, 1952.7634277

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7107504, upper bound: 1781.7107504
time: 0.56 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7107504, upper bound: 1781.7107504
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -279.0773621, 1132.6473389, -371.8542480, 1491.4521484, -1770.5294189, 1504.5015869
1: -449.5133057, 1253.9390869, -601.9755249, 1656.0629883, -2105.5759277, 1855.9144287
2: -335.5559998, 1443.9549561, -449.9300232, 1903.8698730, -2239.4257812, 1893.8847656
3: -722.1193848, 1290.4527588, -969.7329712, 1708.7253418, -2430.8447266, 2260.1857910
4: -578.0726318, 1350.5883789, -774.2708130, 1785.2263184, -2363.2985840, 2124.8591309

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7107504, upper bound: 1781.7107504
time: 0.55 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7107504, upper bound: 1781.7107504
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -276.7585144, 1124.4237061, -399.9865112, 1617.3303223, -1894.0886230, 1524.4101562
1: -446.0162659, 1244.2011719, -643.5168457, 1790.6998291, -2236.7158203, 1887.7178955
2: -332.7811279, 1433.4194336, -480.6738281, 2061.2387695, -2394.0192871, 1914.0932617
3: -715.9465942, 1280.3992920, -1033.8574219, 1845.7706299, -2561.7172852, 2314.2563477
4: -573.1133423, 1340.8247070, -828.6842041, 1929.7155762, -2502.8286133, 2169.5085449

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7128011, upper bound: 1781.7153278
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7128011, upper bound: 1781.7153278
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -371.8542480, 1491.4521484, -388.0945740, 1569.8585205, -1941.7127686, 1879.5467529
1: -601.9755249, 1656.0629883, -624.3645020, 1738.5485840, -2340.5241699, 2280.4267578
2: -449.9300232, 1903.8698730, -466.5751953, 2000.7368164, -2450.6667480, 2370.4450684
3: -969.7329712, 1708.7253418, -1003.5248413, 1791.5582275, -2760.1096191, 2712.2502441
4: -774.2708130, 1785.2263184, -804.6671753, 1872.2998047, -2646.5703125, 2589.8930664

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7080555, upper bound: 1781.7113379
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7064448, upper bound: 1781.7064551
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -380.0692139, 1536.6132812, -281.3028870, 1141.6490479, -1521.7181396, 1817.9161377
1: -611.8825684, 1701.3682861, -453.3091736, 1263.3961182, -1875.2786865, 2154.6767578
2: -457.1301575, 1958.7752686, -338.2549438, 1455.3948975, -1912.5249023, 2297.0300293
3: -982.6779175, 1754.2751465, -727.7110596, 1300.9315186, -2283.6093750, 2481.9858398
4: -788.3563232, 1833.5460205, -582.6524048, 1361.9649658, -2150.3205566, 2416.1984863

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7147013, upper bound: 1781.7098544
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7113379, upper bound: 1781.7080555
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -433.4168091, 1752.8944092, -289.6578979, 1175.1263428, -1608.5429688, 2042.5522461
1: -696.7171021, 1941.2100830, -466.7900391, 1300.5187988, -1997.2358398, 2408.0000000
2: -520.5569458, 2234.4389648, -348.3545227, 1498.1142578, -2018.6711426, 2582.7934570
3: -1120.6684570, 1999.0828857, -749.4802856, 1338.9455566, -2459.6140137, 2748.5632324
4: -897.9285889, 2090.1264648, -600.0190430, 1401.7679443, -2299.6962891, 2690.1452637

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7156866, upper bound: 1781.7101552
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7064551, upper bound: 1781.7064448
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -380.0692139, 1536.6132812, -391.8676453, 1584.3438721, -1964.4128418, 1928.4808350
1: -611.8825684, 1701.3682861, -630.7034912, 1754.1191406, -2366.0017090, 2332.0710449
2: -457.1301575, 1958.7752686, -471.1496582, 2019.4416504, -2476.5717773, 2429.9248047
3: -982.6779175, 1754.2751465, -1013.0391846, 1808.5925293, -2791.2705078, 2767.3137207
4: -788.3563232, 1833.5460205, -812.4279785, 1890.5158691, -2678.8718262, 2645.9741211

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7376298, upper bound: 1781.7376509
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7376298, upper bound: 1781.7379405
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -433.4168091, 1752.8944092, -405.3283691, 1639.3203125, -2072.7370605, 2158.2226562
1: -696.7171021, 1941.2100830, -652.1103516, 1814.6867676, -2511.4038086, 2593.3203125
2: -520.5569458, 2234.4389648, -487.0739746, 2089.1757812, -2609.7324219, 2721.5126953
3: -1120.6684570, 1999.0828857, -1047.5103760, 1870.9151611, -2991.5834961, 3046.5932617
4: -897.9285889, 2090.1264648, -839.7683716, 1955.8560791, -2853.7844238, 2929.8947754

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7156866, upper bound: 1781.7101552
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7064448, upper bound: 1781.7064448
time: 0.59 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.58 seconds
IS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -1781.7107504, upper bound: 1781.7107504
IS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -1781.7107504, upper bound: 1781.7107504
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -1781.7107504, upper bound: 1781.7107504
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -1781.7107504, upper bound: 1781.7107504
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -1781.7128011, upper bound: 1781.7153278
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -1781.7128011, upper bound: 1781.7153278
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -1781.7080555, upper bound: 1781.7113379
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -1781.7064448, upper bound: 1781.7064551
IS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -1781.7147013, upper bound: 1781.7098544
IS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -1781.7113379, upper bound: 1781.7080555
IS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -1781.7156866, upper bound: 1781.7101552
IS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -1781.7064551, upper bound: 1781.7064448
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -1781.7376298, upper bound: 1781.7376509
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -1781.7376298, upper bound: 1781.7379405
IS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -1781.7156866, upper bound: 1781.7101552
IS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -1781.7064448, upper bound: 1781.7064448

## BFS IS instance: IS_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -276.7585144, 1124.4237061, -276.7585144, 1124.4237061, -1401.1822510, 1401.1822510
1: -446.0162659, 1244.2011719, -446.0162659, 1244.2011719, -1690.2174072, 1690.2174072
2: -332.7811279, 1433.4194336, -332.7811279, 1433.4194336, -1766.2005615, 1766.2005615
3: -715.9465942, 1280.3992920, -715.9465942, 1280.3992920, -1996.3458252, 1996.3458252
4: -573.1133423, 1340.8247070, -573.1133423, 1340.8247070, -1913.9379883, 1913.9379883

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7119863, upper bound: 1781.7171819
time: 0.56 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7137575, upper bound: 1781.7180215
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -371.8542480, 1491.4521484, -276.7585144, 1124.4237061, -1496.2779541, 1768.2105713
1: -601.9755249, 1656.0629883, -446.0162659, 1244.2011719, -1846.1766357, 2102.0788574
2: -449.9300232, 1903.8698730, -332.7811279, 1433.4194336, -1883.3493652, 2236.6506348
3: -969.7329712, 1708.7253418, -715.9465942, 1280.3992920, -2250.1323242, 2424.6718750
4: -774.2708130, 1785.2263184, -573.1133423, 1340.8247070, -2115.0954590, 2358.3391113

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7119863, upper bound: 1781.7171819
time: 0.51 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7137575, upper bound: 1781.7180215
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -276.7585144, 1124.4237061, -371.8542480, 1491.4521484, -1768.2105713, 1496.2779541
1: -446.0162659, 1244.2011719, -601.9755249, 1656.0629883, -2102.0786133, 1846.1766357
2: -332.7811279, 1433.4194336, -449.9300232, 1903.8698730, -2236.6506348, 1883.3493652
3: -715.9465942, 1280.3992920, -969.7329712, 1708.7253418, -2424.6718750, 2250.1320801
4: -573.1133423, 1340.8247070, -774.2708130, 1785.2263184, -2358.3391113, 2115.0954590

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7101132, upper bound: 1781.7090632
time: 0.53 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7107188, upper bound: 1781.7107188
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -371.8542480, 1491.4521484, -371.8542480, 1491.4521484, -1863.3063965, 1863.3063965
1: -601.9755249, 1656.0629883, -601.9755249, 1656.0629883, -2258.0385742, 2258.0385742
2: -449.9300232, 1903.8698730, -449.9300232, 1903.8698730, -2353.7998047, 2353.7998047
3: -969.7329712, 1708.7253418, -969.7329712, 1708.7253418, -2678.0043945, 2678.0043945
4: -774.2708130, 1785.2263184, -774.2708130, 1785.2263184, -2559.4968262, 2559.4968262

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7101132, upper bound: 1781.7090632
time: 0.53 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7107188, upper bound: 1781.7107188
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -276.7585144, 1124.4237061, -392.9279480, 1589.2556152, -1866.0140381, 1517.3516846
1: -446.0162659, 1244.2011719, -632.1720581, 1759.5748291, -2205.5908203, 1876.3731689
2: -332.7811279, 1433.4194336, -472.1139832, 2025.3890381, -2358.1696777, 1905.5334473
3: -715.9465942, 1280.3992920, -1015.6010742, 1813.4028320, -2529.3493652, 2296.0000000
4: -573.1133423, 1340.8247070, -813.8695679, 1896.0518799, -2469.1647949, 2154.6940918

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7217242, upper bound: 1781.7166781
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7159127, upper bound: 1781.7102057
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -276.7585144, 1124.4237061, -484.6770325, 1944.3590088, -2221.1174316, 1609.1004639
1: -446.0162659, 1244.2011719, -783.1220703, 2158.2265625, -2604.2421875, 2027.3229980
2: -332.7811279, 1433.4194336, -585.5126953, 2480.8889160, -2813.6691895, 2018.9321289
3: -715.9465942, 1280.3992920, -1261.0256348, 2226.6118164, -2942.5583496, 2541.4248047
4: -573.1133423, 1340.8247070, -1008.2093506, 2325.2885742, -2898.4018555, 2349.0334473

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7217242, upper bound: 1781.7166781
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7159127, upper bound: 1781.7102057
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -360.4482727, 1444.0756836, -363.3789368, 1468.8037109, -1829.2518311, 1807.4545898
1: -583.7188721, 1603.9960938, -584.7958374, 1627.1131592, -2210.8320312, 2188.7915039
2: -436.3110657, 1843.8334961, -437.0746460, 1872.4071045, -2308.7177734, 2280.9082031
3: -940.3057861, 1655.4703369, -939.8274536, 1676.7879639, -2615.9519043, 2595.2978516
4: -750.8642578, 1729.3269043, -754.0261841, 1751.9262695, -2502.7900391, 2483.3530273

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7080555, upper bound: 1781.7113379
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7080555, upper bound: 1781.7113379
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -368.4189758, 1477.6213379, -413.2039795, 1671.7362061, -2040.1551514, 1890.8253174
1: -596.4194946, 1640.8089600, -664.2153931, 1851.9598389, -2448.3789062, 2305.0239258
2: -445.7901611, 1886.2261963, -496.3112488, 2131.0529785, -2576.8432617, 2382.5373535
3: -960.8133545, 1692.9158936, -1068.9077148, 1905.9295654, -2865.2729492, 2761.8237305
4: -767.1594238, 1768.5831299, -856.2508545, 1992.4289551, -2759.5876465, 2624.8339844

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7064448, upper bound: 1781.7064551
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7064448, upper bound: 1781.7064551
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -366.0052185, 1479.9211426, -273.8366089, 1111.9321289, -1477.9370117, 1753.7576904
1: -589.1425781, 1638.8649902, -441.2750549, 1230.5052490, -1819.6478271, 2080.1398926
2: -440.0267944, 1886.3773193, -329.2231750, 1417.5482178, -1857.5749512, 2215.6005859
3: -946.3379517, 1689.0131836, -708.3670044, 1266.6523438, -2212.9902344, 2397.3801270
4: -758.7982178, 1765.4758301, -567.0626221, 1326.2058105, -2085.0039062, 2332.5380859

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7117041, upper bound: 1781.7056471
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2

### Relational analysis result of IS_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7129088, upper bound: 1781.7081957
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -460.4322510, 1844.2117920, -269.3232422, 1093.1075439, -1553.5397949, 2113.5351562
1: -744.2117920, 2047.9725342, -433.8347778, 1210.1907959, -1954.4025879, 2481.8073730
2: -556.5748291, 2353.6630859, -323.7986755, 1393.6225586, -1950.1972656, 2677.4616699
3: -1198.3090820, 2113.7905273, -696.9123535, 1245.4425049, -2443.2922363, 2810.7028809
4: -958.5090332, 2206.7712402, -557.8638306, 1303.4796143, -2261.9880371, 2764.6342773

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7113379, upper bound: 1781.7080555
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7113379, upper bound: 1781.7080555
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -418.1317444, 1690.9853516, -282.6839600, 1147.4165039, -1565.5482178, 1973.6691895
1: -672.1157227, 1873.0083008, -455.5588989, 1269.8325195, -1941.9481201, 2328.5671387
2: -502.1312866, 2155.4829102, -339.9327393, 1462.8474121, -1964.9783936, 2495.4155273
3: -1081.2238770, 1928.4916992, -731.4055176, 1307.0644531, -2388.2883301, 2659.8972168
4: -866.1642456, 2016.1826172, -585.4931641, 1368.5040283, -2234.6679688, 2601.6757812

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A2_A1_A1

### Relational analysis result of IS_A2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7149216, upper bound: 1781.7090824
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2_A1_A2

### Relational analysis result of IS_A2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7147538, upper bound: 1781.7090091
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -483.0009155, 1938.7708740, -276.2131348, 1120.7633057, -1603.7641602, 2214.9838867
1: -779.0424194, 2151.9399414, -444.9485474, 1240.8820801, -2019.9245605, 2596.8884277
2: -582.8635864, 2473.3574219, -332.1216431, 1428.9313965, -2011.7949219, 2805.4787598
3: -1254.5689697, 2220.6840820, -714.8723145, 1276.8737793, -2531.4428711, 2935.5563965
4: -1004.4208984, 2317.7651367, -572.1535645, 1336.3638916, -2340.7846680, 2889.9187012

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A2_A2_A1

### Relational analysis result of IS_A2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7045871, upper bound: 1781.7048797
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7050634, upper bound: 1781.7050148
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -380.0692139, 1536.6132812, -380.0692139, 1536.6132812, -1916.6823730, 1916.6823730
1: -611.8825684, 1701.3682861, -611.8825684, 1701.3682861, -2313.2507324, 2313.2507324
2: -457.1301575, 1958.7752686, -457.1301575, 1958.7752686, -2415.9055176, 2415.9055176
3: -982.6779175, 1754.2751465, -982.6779175, 1754.2751465, -2736.9521484, 2736.9523926
4: -788.3563232, 1833.5460205, -788.3563232, 1833.5460205, -2621.9020996, 2621.9020996

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7152216, upper bound: 1781.7095375
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7116978, upper bound: 1781.7075707
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -380.0692139, 1536.6132812, -433.4168091, 1752.8944092, -2132.9636230, 1970.0300293
1: -611.8825684, 1701.3682861, -696.7171021, 1941.2100830, -2553.0927734, 2398.0849609
2: -457.1301575, 1958.7752686, -520.5569458, 2234.4389648, -2691.5690918, 2479.3320312
3: -982.6779175, 1754.2751465, -1120.6684570, 1999.0828857, -2981.7607422, 2874.9433594
4: -788.3563232, 1833.5460205, -897.9285889, 2090.1264648, -2878.4824219, 2731.4746094

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7174259, upper bound: 1781.7221358
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7116978, upper bound: 1781.7080555
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -418.1317444, 1690.9853516, -397.5856934, 1607.6989746, -2025.8306885, 2088.5710449
1: -672.1157227, 1873.0083008, -639.5971680, 1779.9958496, -2452.1108398, 2512.6054688
2: -502.1312866, 2155.4829102, -477.6979980, 2048.8933105, -2551.0239258, 2633.1804199
3: -1081.2238770, 1928.4916992, -1027.5708008, 1834.6824951, -2915.9057617, 2956.0620117
4: -866.1642456, 2016.1826172, -823.5815430, 1918.0582275, -2784.2224121, 2839.7641602

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7151475, upper bound: 1781.7095153
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7151475, upper bound: 1781.7101479
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -483.0009155, 1938.7708740, -385.0609741, 1557.7469482, -2040.7478027, 2323.8315430
1: -779.0424194, 2151.9399414, -619.4464111, 1725.0725098, -2504.1149902, 2771.3862305
2: -582.8635864, 2473.3574219, -462.8510437, 1985.2164307, -2568.0800781, 2936.2084961
3: -1254.5689697, 2220.6840820, -995.5993652, 1777.6612549, -3031.5937500, 3216.2834473
4: -1004.4208984, 2317.7651367, -798.2792969, 1857.6907959, -2862.1115723, 3116.0444336

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7055910, upper bound: 1781.7055910
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2

### Relational analysis result of IS_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7055910, upper bound: 1781.7064448
time: 0.58 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.05 seconds
IS_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 0, lower bound: -1781.7119863, upper bound: 1781.7171819
IS_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 0, lower bound: -1781.7137575, upper bound: 1781.7180215
IS_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 0, lower bound: -1781.7119863, upper bound: 1781.7171819
IS_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 0, lower bound: -1781.7137575, upper bound: 1781.7180215
IS_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 0, lower bound: -1781.7101132, upper bound: 1781.7090632
IS_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 0, lower bound: -1781.7107188, upper bound: 1781.7107188
IS_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 0, lower bound: -1781.7101132, upper bound: 1781.7090632
IS_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 0, lower bound: -1781.7107188, upper bound: 1781.7107188
IS_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 0, lower bound: -1781.7217242, upper bound: 1781.7166781
IS_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 0, lower bound: -1781.7159127, upper bound: 1781.7102057
IS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 0, lower bound: -1781.7217242, upper bound: 1781.7166781
IS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 0, lower bound: -1781.7159127, upper bound: 1781.7102057
IS_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 0, lower bound: -1781.7080555, upper bound: 1781.7113379
IS_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 0, lower bound: -1781.7080555, upper bound: 1781.7113379
IS_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 0, lower bound: -1781.7064448, upper bound: 1781.7064551
IS_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 0, lower bound: -1781.7064448, upper bound: 1781.7064551
IS_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 0, lower bound: -1781.7117041, upper bound: 1781.7056471
IS_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 0, lower bound: -1781.7129088, upper bound: 1781.7081957
IS_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 0, lower bound: -1781.7113379, upper bound: 1781.7080555
IS_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 0, lower bound: -1781.7113379, upper bound: 1781.7080555
IS_A2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 0, lower bound: -1781.7149216, upper bound: 1781.7090824
IS_A2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 0, lower bound: -1781.7147538, upper bound: 1781.7090091
IS_A2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 0, lower bound: -1781.7045871, upper bound: 1781.7048797
IS_A2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 0, lower bound: -1781.7050634, upper bound: 1781.7050148
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 0, lower bound: -1781.7152216, upper bound: 1781.7095375
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 0, lower bound: -1781.7116978, upper bound: 1781.7075707
IS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 0, lower bound: -1781.7174259, upper bound: 1781.7221358
IS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 0, lower bound: -1781.7116978, upper bound: 1781.7080555
IS_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 0, lower bound: -1781.7151475, upper bound: 1781.7095153
IS_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 0, lower bound: -1781.7151475, upper bound: 1781.7101479
IS_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 0, lower bound: -1781.7055910, upper bound: 1781.7055910
IS_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 0, lower bound: -1781.7055910, upper bound: 1781.7064448

## BFS IS instance: IS_A1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -274.4427185, 1115.0172119, -275.7110901, 1120.1723633, -1394.6149902, 1390.7282715
1: -442.2922974, 1233.8166504, -444.3323975, 1239.5045166, -1681.7965088, 1678.1490479
2: -329.9755554, 1421.4028320, -331.5128479, 1427.9886475, -1757.9641113, 1752.9155273
3: -709.9766235, 1269.6837158, -713.2451172, 1275.5546875, -1985.5311279, 1982.9287109
4: -568.2381592, 1329.5943604, -570.9098511, 1335.7495117, -1903.9875488, 1900.5041504

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7362713, upper bound: 1781.7360371
time: 0.76 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7370994, upper bound: 1781.7362680
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -279.3706665, 1134.2976074, -275.3964539, 1119.0139160, -1398.3845215, 1409.6940918
1: -450.4078979, 1255.0859375, -443.7819519, 1238.1672363, -1688.5750732, 1698.8679199
2: -336.0948792, 1446.0902100, -331.1366272, 1426.5151367, -1762.6099854, 1777.2265625
3: -722.8035278, 1291.5921631, -712.3701782, 1274.1204834, -1996.9239502, 2003.9621582
4: -578.5453491, 1352.6071777, -570.2812500, 1334.3035889, -1912.8488770, 1922.8883057

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7354895, upper bound: 1781.7358324
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7360445, upper bound: 1781.7360445
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -369.6397400, 1482.5614014, -275.7110901, 1120.1723633, -1489.8118896, 1758.2724609
1: -598.4256592, 1646.1999512, -444.3323975, 1239.5045166, -1837.9301758, 2090.5322266
2: -447.2619019, 1892.4997559, -331.5128479, 1427.9886475, -1875.2503662, 2224.0124512
3: -964.0110474, 1698.5567627, -713.2451172, 1275.5546875, -2239.5656738, 2411.8017578
4: -769.6611938, 1774.5987549, -570.9098511, 1335.7495117, -2105.4106445, 2345.5083008

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7119863, upper bound: 1781.7171819
time: 0.79 seconds

## Relational analysis of IS_A1_B1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7119863, upper bound: 1781.7171819
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -369.2091370, 1480.5931396, -275.3964539, 1119.0139160, -1488.2229004, 1755.9893799
1: -597.6428223, 1644.0800781, -443.7819519, 1238.1672363, -1835.8100586, 2087.8620605
2: -446.7429504, 1890.0368652, -331.1366272, 1426.5151367, -1873.2580566, 2221.1730957
3: -962.7859497, 1696.2437744, -712.3701782, 1274.1204834, -2236.9064941, 2408.6140137
4: -768.7138062, 1772.2023926, -570.2812500, 1334.3035889, -2103.0173340, 2342.4836426

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7137575, upper bound: 1781.7180215
time: 0.57 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7137575, upper bound: 1781.7180215
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -275.7110901, 1120.1723633, -369.6397400, 1482.5614014, -1758.2724609, 1489.8118896
1: -444.3323975, 1239.5045166, -598.4256592, 1646.1999512, -2090.5322266, 1837.9301758
2: -331.5128479, 1427.9886475, -447.2619019, 1892.4997559, -2224.0124512, 1875.2504883
3: -713.2451172, 1275.5546875, -964.0110474, 1698.5567627, -2411.8017578, 2239.5656738
4: -570.9098511, 1335.7495117, -769.6611938, 1774.5987549, -2345.5085449, 2105.4106445

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7171819, upper bound: 1781.7119863
time: 0.58 seconds

## Relational analysis of IS_A1_B1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7171819, upper bound: 1781.7119863
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -275.3964539, 1119.0139160, -369.2091370, 1480.5931396, -1755.9893799, 1488.2229004
1: -443.7819519, 1238.1672363, -597.6428223, 1644.0800781, -2087.8620605, 1835.8100586
2: -331.1366272, 1426.5151367, -446.7429504, 1890.0368652, -2221.1730957, 1873.2580566
3: -712.3701782, 1274.1204834, -962.7859497, 1696.2437744, -2408.6140137, 2236.9064941
4: -570.2812500, 1334.3035889, -768.7138062, 1772.2023926, -2342.4836426, 2103.0173340

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7180215, upper bound: 1781.7137575
time: 0.53 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7180215, upper bound: 1781.7137575
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -370.8994141, 1487.6170654, -369.6397400, 1482.5614014, -1853.4606934, 1857.2565918
1: -600.4448242, 1651.8092041, -598.4256592, 1646.1999512, -2246.6445312, 2250.2346191
2: -448.7796021, 1898.9655762, -447.2619019, 1892.4997559, -2341.2792969, 2346.2275391
3: -967.2656860, 1704.3400879, -964.0110474, 1698.5567627, -2665.3876953, 2667.9172363
4: -772.2833862, 1780.6430664, -769.6611938, 1774.5987549, -2546.8815918, 2550.3041992

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7084579, upper bound: 1781.7084579
time: 0.58 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7084579, upper bound: 1781.7090632
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -370.3101196, 1485.2576904, -369.2091370, 1480.5931396, -1850.9031982, 1854.4665527
1: -599.4796143, 1649.2341309, -597.6428223, 1644.0800781, -2243.5595703, 2246.8762207
2: -448.0728149, 1895.9647217, -446.7429504, 1890.0368652, -2338.1093750, 2342.7077637
3: -965.7340088, 1701.6087646, -962.7859497, 1696.2437744, -2661.7319336, 2664.0148926
4: -771.0535889, 1777.7799072, -768.7138062, 1772.2023926, -2543.2558594, 2546.4934082

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7090632, upper bound: 1781.7101132
time: 0.58 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7090632, upper bound: 1781.7107188
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -265.5378418, 1079.1926270, -366.0052185, 1479.9211426, -1745.4588623, 1445.1975098
1: -427.9176025, 1194.1444092, -589.1425781, 1638.8649902, -2066.7819824, 1783.2869873
2: -319.1988831, 1375.7774658, -440.0267944, 1886.3773193, -2205.5761719, 1815.8041992
3: -686.9018555, 1228.6964111, -946.3379517, 1689.0131836, -2375.9150391, 2175.0344238
4: -549.7484741, 1286.7053223, -758.7982178, 1765.4758301, -2315.2238770, 2045.5035400

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7350869, upper bound: 1781.7342895
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7348085, upper bound: 1781.7344001
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -274.6969910, 1115.8557129, -417.2734680, 1687.4654541, -1962.1624756, 1533.1290283
1: -442.7150269, 1234.7864990, -670.7286987, 1869.1072998, -2311.8215332, 1905.5151367
2: -330.3060608, 1422.6062012, -501.1014709, 2150.9934082, -2481.2990723, 1923.7076416
3: -710.7012939, 1270.5880127, -1078.9874268, 1924.5645752, -2635.2658691, 2349.5751953
4: -568.8641357, 1330.5485840, -864.3917847, 2012.0499268, -2580.9135742, 2194.9394531

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7347726, upper bound: 1781.7351038
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7348715, upper bound: 1781.7346886
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -265.5378418, 1079.1926270, -460.4322510, 1844.2117920, -2109.7495117, 1539.6247559
1: -427.9176025, 1194.1444092, -744.2117920, 2047.9725342, -2475.8901367, 1938.3562012
2: -319.1988831, 1375.7774658, -556.5748291, 2353.6630859, -2672.8620605, 1932.3522949
3: -686.9018555, 1228.6964111, -1198.3090820, 2113.7905273, -2800.6923828, 2426.7778320
4: -549.7484741, 1286.7053223, -958.5090332, 2206.7712402, -2756.5190430, 2245.2136230

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7194714, upper bound: 1781.7143462
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7166198, upper bound: 1781.7097669
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -274.6969910, 1115.8557129, -483.0009155, 1938.7708740, -2213.4677734, 1598.8566895
1: -442.7150269, 1234.7864990, -779.0424194, 2151.9399414, -2594.6550293, 2013.8288574
2: -330.3060608, 1422.6062012, -582.8635864, 2473.3574219, -2803.6633301, 2005.4697266
3: -710.7012939, 1270.5880127, -1254.5689697, 2220.6840820, -2931.3850098, 2525.1564941
4: -568.8641357, 1330.5485840, -1004.4208984, 2317.7651367, -2886.6291504, 2334.9694824

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7153961, upper bound: 1781.7089042
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7154175, upper bound: 1781.7093405
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -360.4482727, 1444.0756836, -364.9979858, 1475.9707031, -1836.4188232, 1809.0736084
1: -583.7188721, 1603.9960938, -587.5361328, 1634.4697266, -2218.1884766, 2191.5314941
2: -436.3110657, 1843.8334961, -438.8081665, 1881.3377686, -2317.6486816, 2282.6416016
3: -940.3057861, 1655.4703369, -943.7641602, 1684.4112549, -2623.8710938, 2599.2343750
4: -750.8642578, 1729.3269043, -756.7068481, 1760.6683350, -2511.5327148, 2486.0336914

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7080431, upper bound: 1781.7111361
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7027228, upper bound: 1781.7039521
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7052296, upper bound: 1781.7050582
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -360.4482727, 1444.0756836, -460.4322510, 1844.2117920, -2204.6599121, 1904.5079346
1: -583.7188721, 1603.9960938, -744.2117920, 2047.9725342, -2631.6914062, 2348.2080078
2: -436.3110657, 1843.8334961, -556.5748291, 2353.6630859, -2789.9741211, 2400.4079590
3: -940.3057861, 1655.4703369, -1198.3090820, 2113.7905273, -3051.7111816, 2852.7482910
4: -750.8642578, 1729.3269043, -958.5090332, 2206.7712402, -2957.6347656, 2687.8352051

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7062673, upper bound: 1781.7095760
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7080431, upper bound: 1781.7111361
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7027228, upper bound: 1781.7039521
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7052296, upper bound: 1781.7050582
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -368.4189758, 1477.6213379, -415.8037415, 1681.8759766, -2050.2949219, 1893.4250488
1: -596.4194946, 1640.8089600, -668.4752197, 1862.8389893, -2459.2580566, 2309.2839355
2: -445.7901611, 1886.2261963, -499.3606262, 2143.8862305, -2589.6762695, 2385.5866699
3: -960.8133545, 1692.9158936, -1075.3530273, 1917.9638672, -2877.4763184, 2768.2687988
4: -767.1594238, 1768.5831299, -861.4062500, 2005.1766357, -2772.3359375, 2629.9892578

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7025294, upper bound: 1781.7040195
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7050148, upper bound: 1781.7050634
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -368.4189758, 1477.6213379, -483.0009155, 1938.7708740, -2307.1896973, 1960.6223145
1: -596.4194946, 1640.8089600, -779.0424194, 2151.9399414, -2748.3591309, 2419.8513184
2: -445.7901611, 1886.2261963, -582.8635864, 2473.3574219, -2919.1474609, 2469.0898438
3: -960.8133545, 1692.9158936, -1254.5689697, 2220.6840820, -3179.0588379, 2947.4848633
4: -767.1594238, 1768.5831299, -1004.4208984, 2317.7651367, -3084.9243164, 2773.0039062

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7048797, upper bound: 1781.7045871
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2_B2

### Relational analysis result of IS_A1_B2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7050148, upper bound: 1781.7050634
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -364.9170227, 1475.5103760, -271.6943054, 1103.2406006, -1468.1574707, 1747.2047119
1: -587.3840942, 1634.0006104, -437.8262024, 1220.9071045, -1808.2911377, 2071.8269043
2: -438.7158813, 1880.7452393, -326.6294861, 1406.4436035, -1845.1594238, 2207.3747559
3: -943.5238647, 1683.9852295, -702.8305054, 1256.7254639, -2200.2492676, 2386.8156738
4: -756.5307617, 1760.1906738, -562.5581665, 1315.8127441, -2072.3435059, 2322.7487793

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7117041, upper bound: 1781.7056471
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_A1_B1_B2

### Relational analysis result of IS_A2_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7117041, upper bound: 1781.7056471
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -364.7897339, 1474.9896240, -277.0217896, 1124.0216064, -1488.8112793, 1752.0114746
1: -587.1716309, 1633.4006348, -446.5321960, 1243.8632812, -1831.0349121, 2079.9328613
2: -438.5598755, 1880.0843506, -333.1986084, 1433.0729980, -1871.6328125, 2213.2827148
3: -943.1866455, 1683.3779297, -716.6837769, 1280.4783936, -2223.6645508, 2400.0615234
4: -756.2769775, 1759.5854492, -573.6576538, 1340.8140869, -2097.0908203, 2333.2431641

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_A1_B2_B1

### Relational analysis result of IS_A2_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7129088, upper bound: 1781.7081957
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2_B2

### Relational analysis result of IS_A2_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7129088, upper bound: 1781.7081957
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -460.4322510, 1844.2117920, -265.5378418, 1079.1926270, -1539.6247559, 2109.7495117
1: -744.2117920, 2047.9725342, -427.9176025, 1194.1444092, -1938.3562012, 2475.8901367
2: -556.5748291, 2353.6630859, -319.1988831, 1375.7774658, -1932.3522949, 2672.8620605
3: -1198.3090820, 2113.7905273, -686.9018555, 1228.6964111, -2426.7778320, 2800.6923828
4: -958.5090332, 2206.7712402, -549.7484741, 1286.7053223, -2245.2138672, 2756.5190430

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7097506, upper bound: 1781.7066768
time: 0.51 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7050582, upper bound: 1781.7052296
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -460.4322510, 1844.2117920, -360.4482727, 1444.0756836, -1904.5079346, 2204.6599121
1: -744.2117920, 2047.9725342, -583.7188721, 1603.9960938, -2348.2080078, 2631.6914062
2: -556.5748291, 2353.6630859, -436.3110657, 1843.8334961, -2400.4079590, 2789.9741211
3: -1198.3090820, 2113.7905273, -940.3057861, 1655.4703369, -2852.7482910, 3051.7111816
4: -958.5090332, 2206.7712402, -750.8642578, 1729.3269043, -2687.8352051, 2957.6347656

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_A2_B2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7095760, upper bound: 1781.7062673
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_A2_B2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7111361, upper bound: 1781.7080431
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A1_A2_B2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7039521, upper bound: 1781.7027228
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7050582, upper bound: 1781.7052296
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -415.8189392, 1681.6743164, -281.6274414, 1143.1215820, -1558.9403076, 1963.3017578
1: -668.4016724, 1862.7587891, -453.8605347, 1265.0861816, -1933.4877930, 2316.6193848
2: -499.3358154, 2143.6098633, -338.6550598, 1457.3679199, -1956.7036133, 2482.2648926
3: -1075.2943115, 1917.7653809, -728.6799316, 1302.1525879, -2377.4462891, 2646.4453125
4: -861.3146973, 2004.9788818, -583.2730103, 1363.3762207, -2224.6909180, 2588.2519531

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_A1_A1_B1

### Relational analysis result of IS_A2_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7149216, upper bound: 1781.7090824
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2_A1_A1_B2

### Relational analysis result of IS_A2_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7149216, upper bound: 1781.7090824
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -413.8032227, 1673.6517334, -281.3143921, 1142.0162354, -1555.8193359, 1954.9660645
1: -665.5684814, 1853.7893066, -453.3106384, 1263.8138428, -1929.3823242, 2307.0996094
2: -497.0430908, 2133.4689941, -338.2795715, 1455.9415283, -1952.9846191, 2471.7485352
3: -1070.9802246, 1908.5093994, -727.7987671, 1300.8037109, -2371.7836914, 2636.3081055
4: -857.1394043, 1995.6729736, -582.6461792, 1361.9765625, -2219.1157227, 2578.3190918

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_A1_A2_B1

### Relational analysis result of IS_A2_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7147538, upper bound: 1781.7090091
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2_A1_A2_B2

### Relational analysis result of IS_A2_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7147538, upper bound: 1781.7090091
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -480.3002014, 1927.8143311, -275.2664490, 1116.9456787, -1597.2458496, 2203.0808105
1: -774.6972046, 2139.8459473, -443.4284668, 1236.6555176, -2011.3526611, 2583.2744141
2: -579.6039429, 2459.3994141, -330.9777527, 1424.0566406, -2003.6605225, 2790.3769531
3: -1247.5859375, 2208.1982422, -712.4281616, 1272.4947510, -2520.0803223, 2920.6264648
4: -998.7920532, 2304.6918945, -570.1667480, 1331.8026123, -2330.5944824, 2874.8586426

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_A2_A1_B1

### Relational analysis result of IS_A2_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7045871, upper bound: 1781.7048797
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2_A2_A1_B2

### Relational analysis result of IS_A2_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7045871, upper bound: 1781.7048797
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -478.7261353, 1921.3687744, -274.6470642, 1114.4766846, -1593.2028809, 2196.0153809
1: -772.1367798, 2132.7263184, -442.3719788, 1233.9125977, -2006.0493164, 2575.0983887
2: -577.7418213, 2451.1293945, -330.2233276, 1420.9023438, -1998.6441650, 2781.3527832
3: -1243.5155029, 2200.9172363, -710.7583008, 1269.6333008, -2513.1486816, 2911.6750488
4: -995.5452271, 2296.9990234, -568.8761597, 1328.8046875, -2324.3493652, 2865.8752441

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_A2_A2_B1

### Relational analysis result of IS_A2_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7050634, upper bound: 1781.7050148
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A2_A2_A2_B2

### Relational analysis result of IS_A2_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7050634, upper bound: 1781.7050148
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -366.0052185, 1479.9211426, -372.5797424, 1506.1684570, -1872.1735840, 1852.5007324
1: -589.1425781, 1638.8649902, -599.7585449, 1667.9024658, -2257.0449219, 2238.6235352
2: -440.0267944, 1886.3773193, -448.0287476, 1919.9273682, -2359.9541016, 2334.4060059
3: -946.3379517, 1689.0131836, -963.3511353, 1719.2457275, -2665.5837402, 2652.3642578
4: -758.7982178, 1765.4758301, -772.6316528, 1796.9951172, -2555.7934570, 2538.1074219

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7219581, upper bound: 1781.7216495
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7219581, upper bound: 1781.7216495
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -460.4322510, 1844.2117920, -363.3789368, 1468.8037109, -1929.2358398, 2207.5905762
1: -744.2117920, 2047.9725342, -584.7958374, 1627.1131592, -2371.3249512, 2632.7683105
2: -556.5748291, 2353.6630859, -437.0746460, 1872.4071045, -2428.9816895, 2790.7377930
3: -1198.3090820, 2113.7905273, -939.8274536, 1676.7879639, -2873.4694824, 3053.6179199
4: -958.5090332, 2206.7712402, -754.0261841, 1751.9262695, -2710.4345703, 2960.7973633

Time for backsubstitution: 1.43 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=2073.053466796875
rel_dist={0: [-1781.7388006996591, 1781.738800699659]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1136.37 seconds
