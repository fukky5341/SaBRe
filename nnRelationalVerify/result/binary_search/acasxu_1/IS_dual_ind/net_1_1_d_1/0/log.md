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
execution time: IAR + LP analysis = 1.37 + 1.73 = 3.10 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1781.7419244, upper bound: 1781.7419244


# Binary Search by BASE starts (time budget: 1196.90 seconds, max iter: 100)

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
Binary search time: 62.85 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1134.05 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

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
time: 1.95 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.61 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.61
Output dim: 0, lower bound: -1781.7419244, upper bound: 1781.7419244
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.61
Output dim: 0, lower bound: -1781.7419244, upper bound: 1781.7419244

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -292.2078247, 1185.7346191, -409.8881226, 1658.3120117, -1950.5197754, 1595.6226807
1: -470.8653564, 1312.1569824, -659.4754639, 1835.4072266, -2306.2724609, 1971.6322021
2: -351.4121094, 1511.5142822, -492.6187134, 2113.4978027, -2464.9099121, 2004.1329346
3: -755.9409180, 1351.1541748, -1059.2000732, 1892.1563721, -2648.0971680, 2410.3542480
4: -605.2770386, 1414.5281982, -849.2722778, 1978.4670410, -2583.7441406, 2263.8005371

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7419244, upper bound: 1781.7419244
time: 1.93 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7419244, upper bound: 1781.7419244
time: 1.93 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -407.7161560, 1648.8684082, -410.8436279, 1662.2098389, -2069.9260254, 2059.7119141
1: -656.0112915, 1825.3117676, -661.0173340, 1839.7109375, -2495.7221680, 2486.3291016
2: -490.0357056, 2101.4226074, -493.7767944, 2118.4650879, -2608.5007324, 2595.1992188
3: -1053.7666016, 1881.9282227, -1061.6599121, 1896.6071777, -2950.3735352, 2943.5881348
4: -844.8474731, 1967.4259033, -851.2732544, 1983.1082764, -2827.9558105, 2818.6989746

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7419244, upper bound: 1781.7419244
time: 0.53 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7419244, upper bound: 1781.7419244
time: 0.58 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.47 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.47
Output dim: 0, lower bound: -1781.7419244, upper bound: 1781.7419244
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.47
Output dim: 0, lower bound: -1781.7419244, upper bound: 1781.7419244
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.47
Output dim: 0, lower bound: -1781.7419244, upper bound: 1781.7419244
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.47
Output dim: 0, lower bound: -1781.7419244, upper bound: 1781.7419244

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -292.2078247, 1185.7346191, -292.2078247, 1185.7346191, -1477.9423828, 1477.9423828
1: -470.8653564, 1312.1569824, -470.8653564, 1312.1569824, -1783.0222168, 1783.0222168
2: -351.4121094, 1511.5142822, -351.4121094, 1511.5142822, -1862.9263916, 1862.9262695
3: -755.9409180, 1351.1541748, -755.9409180, 1351.1541748, -2107.0952148, 2107.0952148
4: -605.2770386, 1414.5281982, -605.2770386, 1414.5281982, -2019.8050537, 2019.8050537

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7407028, upper bound: 1781.7409421
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7419244, upper bound: 1781.7419244
time: 0.53 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -292.2078247, 1185.7346191, -407.7161560, 1648.8684082, -1941.0761719, 1593.4508057
1: -470.8653564, 1312.1569824, -656.0112915, 1825.3117676, -2296.1772461, 1968.1682129
2: -351.4121094, 1511.5142822, -490.0357056, 2101.4226074, -2452.8347168, 2001.5500488
3: -755.9409180, 1351.1541748, -1053.7666016, 1881.9282227, -2637.8691406, 2404.9206543
4: -605.2770386, 1414.5281982, -844.8474731, 1967.4259033, -2572.7026367, 2259.3757324

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7407028, upper bound: 1781.7409421
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7419244, upper bound: 1781.7419244
time: 0.54 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -407.7161560, 1648.8684082, -292.2078247, 1185.7346191, -1593.4508057, 1941.0761719
1: -656.0112915, 1825.3117676, -470.8653564, 1312.1569824, -1968.1682129, 2296.1772461
2: -490.0357056, 2101.4226074, -351.4121094, 1511.5142822, -2001.5499268, 2452.8347168
3: -1053.7666016, 1881.9282227, -755.9409180, 1351.1541748, -2404.9206543, 2637.8691406
4: -844.8474731, 1967.4259033, -605.2770386, 1414.5281982, -2259.3757324, 2572.7026367

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

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
time: 0.54 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -407.7161560, 1648.8684082, -407.7161560, 1648.8684082, -2056.5844727, 2056.5844727
1: -656.0112915, 1825.3117676, -656.0112915, 1825.3117676, -2481.3229980, 2481.3229980
2: -490.0357056, 2101.4226074, -490.0357056, 2101.4226074, -2591.4582520, 2591.4582520
3: -1053.7666016, 1881.9282227, -1053.7666016, 1881.9282227, -2935.6948242, 2935.6948242
4: -844.8474731, 1967.4259033, -844.8474731, 1967.4259033, -2812.2729492, 2812.2729492

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7403250, upper bound: 1781.7404735
time: 0.50 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7405927, upper bound: 1781.7405927
time: 0.52 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.37 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.37
Output dim: 0, lower bound: -1781.7407028, upper bound: 1781.7409421
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.37
Output dim: 0, lower bound: -1781.7419244, upper bound: 1781.7419244
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.37
Output dim: 0, lower bound: -1781.7407028, upper bound: 1781.7409421
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.37
Output dim: 0, lower bound: -1781.7419244, upper bound: 1781.7419244
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.37
Output dim: 0, lower bound: -1781.7403250, upper bound: 1781.7404735
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.37
Output dim: 0, lower bound: -1781.7405927, upper bound: 1781.7405927
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.37
Output dim: 0, lower bound: -1781.7403250, upper bound: 1781.7404735
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.37
Output dim: 0, lower bound: -1781.7405927, upper bound: 1781.7405927

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -255.5607452, 1036.6025391, -290.3953857, 1178.3992920, -1433.9599609, 1326.9975586
1: -411.7577515, 1147.1502686, -467.8760681, 1304.0753174, -1715.8328857, 1615.0263672
2: -306.8365173, 1321.8051758, -349.1882629, 1502.1998291, -1809.0363770, 1670.9934082
3: -661.7048950, 1180.2808838, -751.2254028, 1342.8186035, -2004.5234375, 1931.5062256
4: -528.7719727, 1236.8221436, -601.5274048, 1405.7766113, -1934.5485840, 1838.3494873

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7397205, upper bound: 1781.7397205
time: 0.47 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7397205, upper bound: 1781.7409421
time: 0.47 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -287.1474915, 1165.4420166, -292.2078247, 1185.7346191, -1472.8820801, 1457.6496582
1: -462.8331604, 1290.0263672, -470.8653564, 1312.1569824, -1774.9901123, 1760.8917236
2: -345.4135437, 1485.2845459, -351.4121094, 1511.5142822, -1856.9276123, 1836.6966553
3: -742.6971436, 1328.7269287, -755.9409180, 1351.1541748, -2093.8510742, 2084.6677246
4: -595.0137329, 1390.4621582, -605.2770386, 1414.5281982, -2009.5418701, 1995.7391357

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7409421, upper bound: 1781.7407028
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7409421, upper bound: 1781.7419244
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -255.5607452, 1036.6025391, -405.7738647, 1640.9945068, -1896.5552979, 1442.3760986
1: -411.7577515, 1147.1502686, -652.8298340, 1816.6247559, -2228.3825684, 1799.9801025
2: -306.8365173, 1321.8051758, -487.6717529, 2091.4101562, -2398.2465820, 1809.4769287
3: -661.7048950, 1180.2808838, -1048.7415771, 1873.0562744, -2534.7609863, 2229.0222168
4: -528.7719727, 1236.8221436, -840.8638306, 1958.0939941, -2486.8659668, 2077.6857910

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7391311, upper bound: 1781.7393290
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7392503, upper bound: 1781.7395968
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -287.1474915, 1165.4420166, -407.7161560, 1648.8684082, -1936.0158691, 1573.1579590
1: -462.8331604, 1290.0263672, -656.0112915, 1825.3117676, -2288.1450195, 1946.0375977
2: -345.4135437, 1485.2845459, -490.0357056, 2101.4226074, -2446.8361816, 1975.3203125
3: -742.6971436, 1328.7269287, -1053.7666016, 1881.9282227, -2624.6254883, 2382.4931641
4: -595.0137329, 1390.4621582, -844.8474731, 1967.4259033, -2562.4396973, 2235.3095703

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7404735, upper bound: 1781.7403250
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7405927, upper bound: 1781.7405927
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -380.0692139, 1536.6132812, -292.2078247, 1185.7346191, -1565.8037109, 1828.8210449
1: -611.8825684, 1701.3682861, -470.8653564, 1312.1569824, -1924.0395508, 2172.2331543
2: -457.1301575, 1958.7752686, -351.4121094, 1511.5142822, -1968.6442871, 2310.1875000
3: -982.6779175, 1754.2751465, -755.9409180, 1351.1541748, -2333.8315430, 2510.2158203
4: -788.3563232, 1833.5460205, -605.2770386, 1414.5281982, -2202.8842773, 2438.8229980

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7393290, upper bound: 1781.7391311
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7403250, upper bound: 1781.7404735
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -433.4168091, 1752.8944092, -292.0651245, 1185.1374512, -1618.5541992, 2044.9594727
1: -696.7171021, 1941.2100830, -470.6372681, 1311.5079346, -2008.2249756, 2411.8474121
2: -520.5569458, 2234.4389648, -351.2402954, 1510.7570801, -2031.3139648, 2585.6791992
3: -1120.6684570, 1999.0828857, -755.5812988, 1350.4630127, -2471.1313477, 2754.6640625
4: -897.9285889, 2090.1264648, -604.9828491, 1413.8060303, -2311.7346191, 2695.1093750

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7395968, upper bound: 1781.7392503
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7405927, upper bound: 1781.7405927
time: 0.48 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -380.0692139, 1536.6132812, -407.7161560, 1648.8684082, -2028.9375000, 1944.3294678
1: -611.8825684, 1701.3682861, -656.0112915, 1825.3117676, -2437.1943359, 2357.3791504
2: -457.1301575, 1958.7752686, -490.0357056, 2101.4226074, -2558.5527344, 2448.8110352
3: -982.6779175, 1754.2751465, -1053.7666016, 1881.9282227, -2864.6062012, 2808.0412598
4: -788.3563232, 1833.5460205, -844.8474731, 1967.4259033, -2755.7817383, 2678.3935547

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7401786, upper bound: 1781.7402057
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7401786, upper bound: 1781.7404735
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -433.4168091, 1752.8944092, -407.6167908, 1648.4715576, -2081.8884277, 2160.5112305
1: -696.7171021, 1941.2100830, -655.8494263, 1824.8702393, -2521.5874023, 2597.0595703
2: -520.5569458, 2234.4389648, -489.9127502, 2100.9128418, -2621.4694824, 2724.3515625
3: -1120.6684570, 1999.0828857, -1053.5073242, 1881.4702148, -3002.1386719, 3052.5903320
4: -897.9285889, 2090.1264648, -844.6366577, 1966.9448242, -2864.8728027, 2934.7631836

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7401786, upper bound: 1781.7402079
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7401786, upper bound: 1781.7405927
time: 0.56 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.58 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -1781.7397205, upper bound: 1781.7397205
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -1781.7397205, upper bound: 1781.7409421
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -1781.7409421, upper bound: 1781.7407028
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -1781.7409421, upper bound: 1781.7419244
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -1781.7391311, upper bound: 1781.7393290
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -1781.7392503, upper bound: 1781.7395968
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -1781.7404735, upper bound: 1781.7403250
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -1781.7405927, upper bound: 1781.7405927
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -1781.7393290, upper bound: 1781.7391311
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -1781.7403250, upper bound: 1781.7404735
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -1781.7395968, upper bound: 1781.7392503
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -1781.7405927, upper bound: 1781.7405927
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -1781.7401786, upper bound: 1781.7402057
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -1781.7401786, upper bound: 1781.7404735
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -1781.7401786, upper bound: 1781.7402079
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -1781.7401786, upper bound: 1781.7405927

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -255.5607452, 1036.6025391, -255.5607452, 1036.6025391, -1292.1632080, 1292.1632080
1: -411.7577515, 1147.1502686, -411.7577515, 1147.1502686, -1558.9079590, 1558.9079590
2: -306.8365173, 1321.8051758, -306.8365173, 1321.8051758, -1628.6417236, 1628.6417236
3: -661.7048950, 1180.2808838, -661.7048950, 1180.2808838, -1841.9858398, 1841.9858398
4: -528.7719727, 1236.8221436, -528.7719727, 1236.8221436, -1765.5941162, 1765.5941162

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7360732, upper bound: 1781.7368072
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7379062, upper bound: 1781.7379062
time: 0.51 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -255.5607452, 1036.6025391, -287.1474915, 1165.4420166, -1421.0026855, 1323.7498779
1: -411.7577515, 1147.1502686, -462.8331604, 1290.0263672, -1701.7841797, 1609.9833984
2: -306.8365173, 1321.8051758, -345.4135437, 1485.2845459, -1792.1210938, 1667.2186279
3: -661.7048950, 1180.2808838, -742.6971436, 1328.7269287, -1990.4318848, 1922.9779053
4: -528.7719727, 1236.8221436, -595.0137329, 1390.4621582, -1919.2341309, 1831.8358154

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7360732, upper bound: 1781.7378231
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7379062, upper bound: 1781.7389222
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -287.1474915, 1165.4420166, -255.5607452, 1036.6025391, -1323.7498779, 1421.0026855
1: -462.8331604, 1290.0263672, -411.7577515, 1147.1502686, -1609.9833984, 1701.7841797
2: -345.4135437, 1485.2845459, -306.8365173, 1321.8051758, -1667.2186279, 1792.1210938
3: -742.6971436, 1328.7269287, -661.7048950, 1180.2808838, -1922.9779053, 1990.4318848
4: -595.0137329, 1390.4621582, -528.7719727, 1236.8221436, -1831.8358154, 1919.2341309

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7382017, upper bound: 1781.7388214
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7383546, upper bound: 1781.7389842
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -287.1474915, 1165.4420166, -287.1474915, 1165.4420166, -1452.5893555, 1452.5893555
1: -462.8331604, 1290.0263672, -462.8331604, 1290.0263672, -1752.8594971, 1752.8594971
2: -345.4135437, 1485.2845459, -345.4135437, 1485.2845459, -1830.6979980, 1830.6979980
3: -742.6971436, 1328.7269287, -742.6971436, 1328.7269287, -2071.4235840, 2071.4235840
4: -595.0137329, 1390.4621582, -595.0137329, 1390.4621582, -1985.4758301, 1985.4758301

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7382017, upper bound: 1781.7391742
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7383546, upper bound: 1781.7393369
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -255.5607452, 1036.6025391, -378.0879517, 1528.6004639, -1784.1612549, 1414.6901855
1: -411.7577515, 1147.1502686, -608.6548462, 1692.5170898, -2104.2749023, 1755.8051758
2: -306.8365173, 1321.8051758, -454.7228088, 1948.5947266, -2255.4311523, 1776.5279541
3: -661.7048950, 1180.2808838, -977.5661621, 1745.1530762, -2406.8579102, 2157.8471680
4: -528.7719727, 1236.8221436, -784.2732544, 1823.9902344, -2352.7622070, 2021.0954590

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7376159, upper bound: 1781.7370068
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7361352, upper bound: 1781.7364283
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7373649, upper bound: 1781.7374924
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -255.3884735, 1035.8767090, -431.4042969, 1744.7846680, -2000.1728516, 1467.2808838
1: -411.4830933, 1146.3613281, -693.4395752, 1932.2501221, -2343.7331543, 1839.8009033
2: -306.6298523, 1320.8872070, -518.1133423, 2224.1276855, -2530.7573242, 1839.0004883
3: -661.2676392, 1179.4519043, -1115.4785156, 1989.8922119, -2651.1599121, 2294.9299316
4: -528.4190063, 1235.9495850, -893.7863159, 2080.4885254, -2608.9072266, 2129.7355957

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7362468, upper bound: 1781.7366710
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7374765, upper bound: 1781.7377351
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -287.1474915, 1165.4420166, -380.0692139, 1536.6132812, -1823.7607422, 1545.5109863
1: -462.8331604, 1290.0263672, -611.8825684, 1701.3682861, -2164.2011719, 1901.9089355
2: -345.4135437, 1485.2845459, -457.1301575, 1958.7752686, -2304.1887207, 1942.4145508
3: -742.6971436, 1328.7269287, -982.6779175, 1754.2751465, -2496.9719238, 2311.4040527
4: -595.0137329, 1390.4621582, -788.3563232, 1833.5460205, -2428.5598145, 2178.8181152

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7375356, upper bound: 1781.7370936
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7377442, upper bound: 1781.7372393
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -286.9993896, 1164.8166504, -433.4168091, 1752.8944092, -2039.8936768, 1598.2332764
1: -462.5976562, 1289.3474121, -696.7171021, 1941.2100830, -2403.8076172, 1986.0644531
2: -345.2360840, 1484.4924316, -520.5569458, 2234.4389648, -2579.6750488, 2005.0493164
3: -742.3257446, 1328.0104980, -1120.6684570, 1999.0828857, -2741.4086914, 2448.6789551
4: -594.7100830, 1389.7104492, -897.9285889, 2090.1264648, -2684.8364258, 2287.6389160

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7376548, upper bound: 1781.7375655
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7378634, upper bound: 1781.7377111
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -378.0879517, 1528.6004639, -255.5607452, 1036.6025391, -1414.6901855, 1784.1612549
1: -608.6548462, 1692.5170898, -411.7577515, 1147.1502686, -1755.8051758, 2104.2749023
2: -454.7228088, 1948.5947266, -306.8365173, 1321.8051758, -1776.5279541, 2255.4311523
3: -977.5661621, 1745.1530762, -661.7048950, 1180.2808838, -2157.8471680, 2406.8579102
4: -784.2732544, 1823.9902344, -528.7719727, 1236.8221436, -2021.0954590, 2352.7622070

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7352323, upper bound: 1781.7380072
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7352323, upper bound: 1781.7391311
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -380.0692139, 1536.6132812, -287.1474915, 1165.4420166, -1545.5109863, 1823.7607422
1: -611.8825684, 1701.3682861, -462.8331604, 1290.0263672, -1901.9089355, 2164.2011719
2: -457.1301575, 1958.7752686, -345.4135437, 1485.2845459, -1942.4145508, 2304.1887207
3: -982.6779175, 1754.2751465, -742.6971436, 1328.7269287, -2311.4040527, 2496.9721680
4: -788.3563232, 1833.5460205, -595.0137329, 1390.4621582, -2178.8181152, 2428.5598145

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7362283, upper bound: 1781.7393496
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7362283, upper bound: 1781.7404735
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -431.4042969, 1744.7846680, -255.3884735, 1035.8767090, -1467.2808838, 2000.1728516
1: -693.4395752, 1932.2501221, -411.4830933, 1146.3613281, -1839.8009033, 2343.7331543
2: -518.1133423, 2224.1276855, -306.6298523, 1320.8872070, -1839.0004883, 2530.7573242
3: -1115.4785156, 1989.8922119, -661.2676392, 1179.4519043, -2294.9301758, 2651.1599121
4: -893.7863159, 2080.4885254, -528.4190063, 1235.9495850, -2129.7355957, 2608.9072266

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7366589, upper bound: 1781.7374567
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7367152, upper bound: 1781.7376023
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -433.4168091, 1752.8944092, -286.9993896, 1164.8166504, -1598.2332764, 2039.8936768
1: -696.7171021, 1941.2100830, -462.5976562, 1289.3474121, -1986.0644531, 2403.8076172
2: -520.5569458, 2234.4389648, -345.2360840, 1484.4924316, -2005.0493164, 2579.6750488
3: -1120.6684570, 1999.0828857, -742.3257446, 1328.0104980, -2448.6789551, 2741.4086914
4: -897.9285889, 2090.1264648, -594.7100830, 1389.7104492, -2287.6386719, 2684.8364258

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7376548, upper bound: 1781.7377178
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7377111, upper bound: 1781.7378634
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -380.0692139, 1536.6132812, -380.0692139, 1536.6132812, -1916.6823730, 1916.6823730
1: -611.8825684, 1701.3682861, -611.8825684, 1701.3682861, -2313.2507324, 2313.2507324
2: -457.1301575, 1958.7752686, -457.1301575, 1958.7752686, -2415.9055176, 2415.9055176
3: -982.6779175, 1754.2751465, -982.6779175, 1754.2751465, -2736.9521484, 2736.9523926
4: -788.3563232, 1833.5460205, -788.3563232, 1833.5460205, -2621.9020996, 2621.9020996

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7361091, upper bound: 1781.7390818
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7402026, upper bound: 1781.7402057
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -380.0692139, 1536.6132812, -433.4168091, 1752.8944092, -2132.9636230, 1970.0300293
1: -611.8825684, 1701.3682861, -696.7171021, 1941.2100830, -2553.0927734, 2398.0849609
2: -457.1301575, 1958.7752686, -520.5569458, 2234.4389648, -2691.5690918, 2479.3320312
3: -982.6779175, 1754.2751465, -1120.6684570, 1999.0828857, -2981.7607422, 2874.9433594
4: -788.3563232, 1833.5460205, -897.9285889, 2090.1264648, -2878.4824219, 2731.4746094

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7361091, upper bound: 1781.7393496
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7402026, upper bound: 1781.7404735
time: 0.50 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -433.4168091, 1752.8944092, -380.0692139, 1536.6132812, -1970.0300293, 2132.9636230
1: -696.7171021, 1941.2100830, -611.8825684, 1701.3682861, -2398.0852051, 2553.0927734
2: -520.5569458, 2234.4389648, -457.1301575, 1958.7752686, -2479.3320312, 2691.5690918
3: -1120.6684570, 1999.0828857, -982.6779175, 1754.2751465, -2874.9436035, 2981.7607422
4: -897.9285889, 2090.1264648, -788.3563232, 1833.5460205, -2731.4746094, 2878.4824219

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7372562, upper bound: 1781.7370064
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7371097, upper bound: 1781.7371207
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -433.4168091, 1752.8944092, -433.4168091, 1752.8944092, -2186.3112793, 2186.3112793
1: -696.7171021, 1941.2100830, -696.7171021, 1941.2100830, -2637.9272461, 2637.9272461
2: -520.5569458, 2234.4389648, -520.5569458, 2234.4389648, -2754.9956055, 2754.9956055
3: -1120.6684570, 1999.0828857, -1120.6684570, 1999.0828857, -3119.7514648, 3119.7514648
4: -897.9285889, 2090.1264648, -897.9285889, 2090.1264648, -2988.0549316, 2988.0549316

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7372562, upper bound: 1781.7375634
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7371097, upper bound: 1781.7377111
time: 0.57 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.60 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7360732, upper bound: 1781.7368072
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7379062, upper bound: 1781.7379062
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7360732, upper bound: 1781.7378231
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7379062, upper bound: 1781.7389222
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7382017, upper bound: 1781.7388214
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7383546, upper bound: 1781.7389842
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7382017, upper bound: 1781.7391742
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7383546, upper bound: 1781.7393369
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7361352, upper bound: 1781.7364283
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7373649, upper bound: 1781.7374924
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7362468, upper bound: 1781.7366710
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7374765, upper bound: 1781.7377351
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7375356, upper bound: 1781.7370936
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7377442, upper bound: 1781.7372393
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7376548, upper bound: 1781.7375655
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7378634, upper bound: 1781.7377111
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7352323, upper bound: 1781.7380072
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7352323, upper bound: 1781.7391311
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7362283, upper bound: 1781.7393496
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7362283, upper bound: 1781.7404735
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7366589, upper bound: 1781.7374567
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7367152, upper bound: 1781.7376023
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7376548, upper bound: 1781.7377178
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7377111, upper bound: 1781.7378634
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7361091, upper bound: 1781.7390818
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7402026, upper bound: 1781.7402057
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7361091, upper bound: 1781.7393496
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7402026, upper bound: 1781.7404735
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7372562, upper bound: 1781.7370064
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7371097, upper bound: 1781.7371207
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7372562, upper bound: 1781.7375634
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7371097, upper bound: 1781.7377111

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -242.2535858, 982.2024536, -254.8965454, 1033.8952637, -1276.1488037, 1237.0989990
1: -390.3833923, 1087.1320801, -410.6863098, 1144.1622314, -1534.5456543, 1497.8183594
2: -290.8264771, 1252.6706543, -306.0412598, 1318.3601074, -1609.1865234, 1558.7119141
3: -627.6691895, 1117.2794189, -659.9959717, 1177.1671143, -1804.8361816, 1777.2753906
4: -500.9706116, 1171.2468262, -527.3930664, 1233.5726318, -1734.5432129, 1698.6398926

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7349742, upper bound: 1781.7349742
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7349742, upper bound: 1781.7368072
time: 0.51 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -253.0717621, 1026.6875000, -255.5607452, 1036.6025391, -1289.6740723, 1282.2480469
1: -407.7406311, 1136.2047119, -411.7577515, 1147.1502686, -1554.8908691, 1547.9624023
2: -303.8264465, 1309.1923828, -306.8365173, 1321.8051758, -1625.6315918, 1616.0289307
3: -655.2677612, 1168.8730469, -661.7048950, 1180.2808838, -1835.5485840, 1830.5778809
4: -523.6318359, 1224.8913574, -528.7719727, 1236.8221436, -1760.4539795, 1753.6633301

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7368072, upper bound: 1781.7360732
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7368072, upper bound: 1781.7379062
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -242.2535858, 982.2024536, -286.4945374, 1162.7777100, -1405.0312500, 1268.6968994
1: -390.3833923, 1087.1320801, -461.7802734, 1287.0871582, -1677.4705811, 1548.9123535
2: -290.8264771, 1252.6706543, -344.6307678, 1481.8901367, -1772.7165527, 1597.3013916
3: -627.6691895, 1117.2794189, -741.0175171, 1325.6589355, -1953.3281250, 1858.2968750
4: -500.9706116, 1171.2468262, -593.6542358, 1387.2642822, -1888.2348633, 1764.9011230

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7350761, upper bound: 1781.7348478
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7352790, upper bound: 1781.7352294
time: 0.51 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -253.0717621, 1026.6875000, -287.1474915, 1165.4420166, -1418.5135498, 1313.8348389
1: -407.7406311, 1136.2047119, -462.8331604, 1290.0263672, -1697.7669678, 1599.0378418
2: -303.8264465, 1309.1923828, -345.4135437, 1485.2845459, -1789.1109619, 1654.6058350
3: -655.2677612, 1168.8730469, -742.6971436, 1328.7269287, -1983.9946289, 1911.5701904
4: -523.6318359, 1224.8913574, -595.0137329, 1390.4621582, -1914.0939941, 1819.9050293

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7368925, upper bound: 1781.7359469
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7370954, upper bound: 1781.7363285
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -272.0064392, 1104.1983643, -255.5607452, 1036.6025391, -1308.6088867, 1359.7591553
1: -438.7653198, 1222.0687256, -411.7577515, 1147.1502686, -1585.9155273, 1633.8264160
2: -327.3316650, 1407.2741699, -306.8365173, 1321.8051758, -1649.1367188, 1714.1105957
3: -704.3140259, 1257.9733887, -661.7048950, 1180.2808838, -1884.5949707, 1919.6782227
4: -563.1935425, 1317.5512695, -528.7719727, 1236.8221436, -1800.0156250, 1846.3232422

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7348478, upper bound: 1781.7350761
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7359469, upper bound: 1781.7368925
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -290.3930359, 1174.6596680, -255.4637451, 1036.2143555, -1326.6071777, 1430.1234131
1: -468.5578918, 1300.0351562, -411.6011353, 1146.7196045, -1615.2773438, 1711.6362305
2: -349.4653015, 1497.5777588, -306.7188721, 1321.3103027, -1670.7756348, 1804.2966309
3: -751.5907593, 1339.8083496, -661.4555054, 1179.8297119, -1931.4200439, 2001.2639160
4: -600.8707275, 1403.1135254, -528.5697632, 1236.3565674, -1837.2271729, 1931.6833496

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7352294, upper bound: 1781.7352790
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7363285, upper bound: 1781.7370954
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -272.0064392, 1104.1983643, -287.1474915, 1165.4420166, -1437.4483643, 1391.3458252
1: -438.7653198, 1222.0687256, -462.8331604, 1290.0263672, -1728.7917480, 1684.9018555
2: -327.3316650, 1407.2741699, -345.4135437, 1485.2845459, -1812.6160889, 1752.6876221
3: -704.3140259, 1257.9733887, -742.6971436, 1328.7269287, -2033.0410156, 2000.6705322
4: -563.1935425, 1317.5512695, -595.0137329, 1390.4621582, -1953.6557617, 1912.5649414

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7388464, upper bound: 1781.7388464
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7388464, upper bound: 1781.7391742
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -290.3930359, 1174.6596680, -287.0690002, 1165.1220703, -1455.5148926, 1461.7286377
1: -468.5578918, 1300.0351562, -462.7052002, 1289.6724854, -1758.2301025, 1762.7402344
2: -349.4653015, 1497.5777588, -345.3184814, 1484.8779297, -1834.3432617, 1842.8961182
3: -751.5907593, 1339.8083496, -742.4960327, 1328.3563232, -2079.9467773, 2082.3044434
4: -600.8707275, 1403.1135254, -594.8508911, 1390.0803223, -1990.9510498, 1997.9643555

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7388464, upper bound: 1781.7388635
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7388464, upper bound: 1781.7393369
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -242.2535858, 982.2024536, -377.5876770, 1526.5290527, -1768.7825928, 1359.7900391
1: -390.3833923, 1087.1320801, -607.8450928, 1690.2415771, -2080.6250000, 1694.9771729
2: -290.8264771, 1252.6706543, -454.1164551, 1945.9566650, -2236.7832031, 1706.7871094
3: -627.6691895, 1117.2794189, -976.2804565, 1742.7791748, -2370.4482422, 2093.5598145
4: -500.9706116, 1171.2468262, -783.2211304, 1821.5142822, -2322.4843750, 1954.4680176

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7350516, upper bound: 1781.7330078
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7350516, upper bound: 1781.7364283
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -253.0717621, 1026.6875000, -378.0879517, 1528.6004639, -1781.6722412, 1404.7751465
1: -407.7406311, 1136.2047119, -608.6548462, 1692.5170898, -2100.2575684, 1744.8593750
2: -303.8264465, 1309.1923828, -454.7228088, 1948.5947266, -2252.4206543, 1763.9150391
3: -655.2677612, 1168.8730469, -977.5661621, 1745.1530762, -2400.4208984, 2146.4392090
4: -523.6318359, 1224.8913574, -784.2732544, 1823.9902344, -2347.6218262, 2009.1645508

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7362813, upper bound: 1781.7340719
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7362813, upper bound: 1781.7374924
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -242.0127258, 981.2075806, -430.8943176, 1742.6911621, -1984.7038574, 1412.1019287
1: -389.9958191, 1086.0404053, -692.6116943, 1929.9443359, -2319.9396973, 1778.6519775
2: -290.5358582, 1251.4107666, -517.4992676, 2221.4606934, -2511.9965820, 1768.9100342
3: -627.0516968, 1116.1368408, -1114.1579590, 1987.4979248, -2614.5495605, 2230.2949219
4: -500.4738464, 1170.0587158, -892.7250977, 2077.9812012, -2578.4545898, 2062.7834473

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7342837, upper bound: 1781.7334345
time: 0.50 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7344867, upper bound: 1781.7335990
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -252.8373108, 1025.7237549, -431.4042969, 1744.7846680, -1997.6218262, 1457.1280518
1: -407.3605347, 1135.1484375, -693.4395752, 1932.2501221, -2339.6105957, 1828.5880127
2: -303.5417480, 1307.9656982, -518.1133423, 2224.1276855, -2527.6691895, 1826.0791016
3: -654.6620483, 1167.7681885, -1115.4785156, 1989.8922119, -2644.5541992, 2283.2465820
4: -523.1476440, 1223.7237549, -893.7863159, 2080.4885254, -2603.6359863, 2117.5100098

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7354294, upper bound: 1781.7344986
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7356323, upper bound: 1781.7346632
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -272.0064392, 1104.1983643, -380.0692139, 1536.6132812, -1808.6197510, 1484.2675781
1: -438.7653198, 1222.0687256, -611.8825684, 1701.3682861, -2140.1325684, 1833.9512939
2: -327.3316650, 1407.2741699, -457.1301575, 1958.7752686, -2286.1069336, 1864.4041748
3: -704.3140259, 1257.9733887, -982.6779175, 1754.2751465, -2458.5886230, 2240.6508789
4: -563.1935425, 1317.5512695, -788.3563232, 1833.5460205, -2396.7395020, 2105.9067383

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7364117, upper bound: 1781.7351532
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7364117, upper bound: 1781.7370936
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -290.3930359, 1174.6596680, -380.0043640, 1536.3474121, -1826.7402344, 1554.6640625
1: -468.5578918, 1300.0351562, -611.7769775, 1701.0742188, -2169.6318359, 1911.8118896
2: -349.4653015, 1497.5777588, -457.0518799, 1958.4368896, -2307.9020996, 1954.6293945
3: -751.5907593, 1339.8083496, -982.5117798, 1753.9696045, -2505.5600586, 2322.3200684
4: -600.8707275, 1403.1135254, -788.2227783, 1833.2293701, -2434.1000977, 2191.3364258

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7366203, upper bound: 1781.7352988
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7366203, upper bound: 1781.7371761
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -271.8345642, 1103.4819336, -433.4168091, 1752.8944092, -2024.7287598, 1536.8985596
1: -438.4944763, 1221.2904053, -696.7171021, 1941.2100830, -2379.7045898, 1918.0075684
2: -327.1272278, 1406.3658447, -520.5569458, 2234.4389648, -2561.5661621, 1926.9228516
3: -703.8865967, 1257.1531982, -1120.6684570, 1999.0828857, -2702.9689941, 2377.8217773
4: -562.8423462, 1316.6882324, -897.9285889, 2090.1264648, -2652.9685059, 2214.6164551

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7373326, upper bound: 1781.7373326
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7373326, upper bound: 1781.7375655
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -290.1746826, 1173.7753906, -433.3481445, 1752.6160889, -2042.7907715, 1607.1235352
1: -468.2087097, 1299.0659180, -696.6046753, 1940.9016113, -2409.1101074, 1995.6704102
2: -349.2021790, 1496.4548340, -520.4738770, 2234.0847168, -2583.2866211, 2016.9285889
3: -751.0340576, 1338.7947998, -1120.4903564, 1998.7620850, -2749.7949219, 2459.2844238
4: -600.4206543, 1402.0515137, -897.7866211, 2089.7934570, -2690.2138672, 2299.8378906

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7377178, upper bound: 1781.7374782
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7377178, upper bound: 1781.7377111
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -340.4019470, 1375.2512207, -255.5607452, 1036.6025391, -1377.0042725, 1630.8120117
1: -548.2256470, 1522.8673096, -411.7577515, 1147.1502686, -1695.3759766, 1934.6248779
2: -409.0312500, 1753.7003174, -306.8365173, 1321.8051758, -1730.8364258, 2060.5368652
3: -881.0024414, 1569.2692871, -661.7048950, 1180.2808838, -2061.2829590, 2230.9741211
4: -705.7205811, 1641.2520752, -528.7719727, 1236.8221436, -1942.5426025, 2170.0239258

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7302320, upper bound: 1781.7363617
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7330078, upper bound: 1781.7350516
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7340719, upper bound: 1781.7362813
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -373.8887634, 1511.7923584, -255.5607452, 1036.6025391, -1410.4913330, 1767.3531494
1: -602.0303955, 1674.2281494, -411.7577515, 1147.1502686, -1749.1806641, 2085.9858398
2: -449.8215332, 1926.8049316, -306.8365173, 1321.8051758, -1771.6267090, 2233.6413574
3: -966.5252686, 1726.8314209, -661.7048950, 1180.2808838, -2146.8059082, 2388.5363770
4: -775.8743896, 1804.1857910, -528.7719727, 1236.8221436, -2012.6961670, 2332.9577637

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7302320, upper bound: 1781.7376159
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7330078, upper bound: 1781.7361352
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7340719, upper bound: 1781.7373649
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -340.4019470, 1375.2512207, -287.1474915, 1165.4420166, -1505.8437500, 1662.3986816
1: -548.2256470, 1522.8673096, -462.8331604, 1290.0263672, -1838.2519531, 1985.7004395
2: -409.0312500, 1753.7003174, -345.4135437, 1485.2845459, -1894.3157959, 2099.1137695
3: -881.0024414, 1569.2692871, -742.6971436, 1328.7269287, -2209.7290039, 2311.9663086
4: -705.7205811, 1641.2520752, -595.0137329, 1390.4621582, -2096.1826172, 2236.2658691

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7302320, upper bound: 1781.7377971
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7330078, upper bound: 1781.7370836
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7340719, upper bound: 1781.7375275
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -373.8887634, 1511.7923584, -287.1474915, 1165.4420166, -1539.3308105, 1798.9398193
1: -602.0303955, 1674.2281494, -462.8331604, 1290.0263672, -1892.0566406, 2137.0612793
2: -449.8215332, 1926.8049316, -345.4135437, 1485.2845459, -1935.1060791, 2272.2185059
3: -966.5252686, 1726.8314209, -742.6971436, 1328.7269287, -2295.2514648, 2469.5285645
4: -775.8743896, 1804.1857910, -595.0137329, 1390.4621582, -2166.3364258, 2399.1994629

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7302320, upper bound: 1781.7387052
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7330078, upper bound: 1781.7375445
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7340719, upper bound: 1781.7383981
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -414.8493958, 1677.3507080, -255.3884735, 1035.8767090, -1450.7260742, 1932.7391357
1: -666.9367065, 1857.9216309, -411.4830933, 1146.3613281, -1813.2980957, 2269.4045410
2: -498.2525940, 2138.1972656, -306.6298523, 1320.8872070, -1819.1397705, 2444.8271484
3: -1073.3891602, 1912.0555420, -661.2676392, 1179.4519043, -2252.8408203, 2573.3232422
4: -859.1561890, 1999.8604736, -528.4190063, 1235.9495850, -2095.1057129, 2528.2790527

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7334345, upper bound: 1781.7342837
time: 0.50 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7344986, upper bound: 1781.7354294
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -430.3157654, 1737.9997559, -255.2917786, 1035.4897461, -1465.8055420, 1993.2915039
1: -692.1263428, 1925.3264160, -411.3269348, 1145.9318848, -1838.0582275, 2336.6533203
2: -516.8775024, 2215.7617188, -306.5126038, 1320.3942871, -1837.2714844, 2522.2744141
3: -1113.5946045, 1982.9691162, -661.0190430, 1179.0023193, -2292.5966797, 2643.9882812
4: -891.1426392, 2073.6633301, -528.2173462, 1235.4855957, -2126.6281738, 2601.8801270

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7335990, upper bound: 1781.7344867
time: 0.50 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7346632, upper bound: 1781.7356323
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -416.8405151, 1685.3878174, -286.9993896, 1164.8166504, -1581.6568604, 1972.3872070
1: -670.1839600, 1866.7962646, -462.5976562, 1289.3474121, -1959.5313721, 2329.3937988
2: -500.6704102, 2148.4094238, -345.2360840, 1484.4924316, -1985.1627197, 2493.6455078
3: -1078.5261230, 1921.1632080, -742.3257446, 1328.0104980, -2406.5366211, 2663.4882812
4: -863.2470093, 2009.3989258, -594.7100830, 1389.7104492, -2252.9572754, 2604.1088867

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7373326, upper bound: 1781.7373326
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7373326, upper bound: 1781.7377178
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -432.3953552, 1746.4383545, -286.9209595, 1164.4974365, -1596.8928223, 2033.3593750
1: -695.5231323, 1934.6020508, -462.4697571, 1288.9943848, -1984.5174561, 2397.0715332
2: -519.4005127, 2226.5065918, -345.1410828, 1484.0864258, -2003.4868164, 2571.6477051
3: -1119.0119629, 1992.5078125, -742.1248169, 1327.6408691, -2446.6523438, 2734.6325684
4: -895.4044189, 2083.7121582, -594.5474243, 1389.3291016, -2284.7329102, 2678.2587891

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7375655, upper bound: 1781.7374782
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7375655, upper bound: 1781.7378634
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -340.4019470, 1375.2512207, -378.0879517, 1528.6004639, -1869.0024414, 1753.3389893
1: -548.2256470, 1522.8673096, -608.6548462, 1692.5170898, -2240.7426758, 2131.5217285
2: -409.0312500, 1753.7003174, -454.7228088, 1948.5947266, -2357.6259766, 2208.4230957
3: -881.0024414, 1569.2692871, -977.5661621, 1745.1530762, -2626.1555176, 2546.8354492
4: -705.7205811, 1641.2520752, -784.2732544, 1823.9902344, -2529.7109375, 2425.5253906

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7350667, upper bound: 1781.7361767
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7351796, upper bound: 1781.7359961
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -373.8887634, 1511.7923584, -380.0692139, 1536.6132812, -1910.5020752, 1891.8615723
1: -602.0303955, 1674.2281494, -611.8825684, 1701.3682861, -2303.3984375, 2286.1105957
2: -449.8215332, 1926.8049316, -457.1301575, 1958.7752686, -2408.5966797, 2383.9350586
3: -966.5252686, 1726.8314209, -982.6779175, 1754.2751465, -2720.7995605, 2709.5092773
4: -775.8743896, 1804.1857910, -788.3563232, 1833.5460205, -2609.4201660, 2592.5415039

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7370071, upper bound: 1781.7373007
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7371190, upper bound: 1781.7371201
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -340.4019470, 1375.2512207, -431.4042969, 1744.7846680, -2085.1865234, 1806.6553955
1: -548.2256470, 1522.8673096, -693.4395752, 1932.2501221, -2480.4755859, 2216.3063965
2: -409.0312500, 1753.7003174, -518.1133423, 2224.1276855, -2633.1589355, 2271.8137207
3: -881.0024414, 1569.2692871, -1115.4785156, 1989.8922119, -2870.8945312, 2684.7478027
4: -705.7205811, 1641.2520752, -893.7863159, 2080.4885254, -2786.2089844, 2535.0383301

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7351532, upper bound: 1781.7364117
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7352988, upper bound: 1781.7364679
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -373.8887634, 1511.7923584, -433.4168091, 1752.8944092, -2126.7832031, 1945.2091064
1: -602.0303955, 1674.2281494, -696.7171021, 1941.2100830, -2543.2404785, 2370.9453125
2: -449.8215332, 1926.8049316, -520.5569458, 2234.4389648, -2684.2602539, 2447.3615723
3: -966.5252686, 1726.8314209, -1120.6684570, 1999.0828857, -2965.6081543, 2847.5000000
4: -775.8743896, 1804.1857910, -897.9285889, 2090.1264648, -2866.0007324, 2702.1140137

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7370936, upper bound: 1781.7375356
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7372393, upper bound: 1781.7375919
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -416.8405151, 1685.3878174, -380.0692139, 1536.6132812, -1953.4536133, 2065.4567871
1: -670.1839600, 1866.7962646, -611.8825684, 1701.3682861, -2371.5515137, 2478.6787109
2: -500.6704102, 2148.4094238, -457.1301575, 1958.7752686, -2459.4458008, 2605.5395508
3: -1078.5261230, 1921.1632080, -982.6779175, 1754.2751465, -2832.8012695, 2903.8400879
4: -863.2470093, 2009.3989258, -788.3563232, 1833.5460205, -2696.7929688, 2797.7551270

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7360941, upper bound: 1781.7350575
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7372562, upper bound: 1781.7370064
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -432.3953552, 1746.4383545, -380.0043640, 1536.3474121, -1968.7427979, 2126.4426270
1: -695.5231323, 1934.6020508, -611.7769775, 1701.0742188, -2396.5971680, 2546.3786621
2: -519.4005127, 2226.5065918, -457.0518799, 1958.4368896, -2477.8374023, 2683.5585938
3: -1119.0119629, 1992.5078125, -982.5117798, 1753.9696045, -2872.9814453, 2975.0195312
4: -895.4044189, 2083.7121582, -788.2227783, 1833.2293701, -2728.6337891, 2871.9350586

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7359471, upper bound: 1781.7351668
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7371097, upper bound: 1781.7371207
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -416.8405151, 1685.3878174, -433.4168091, 1752.8944092, -2169.7348633, 2118.8046875
1: -670.1839600, 1866.7962646, -696.7171021, 1941.2100830, -2611.3940430, 2563.5134277
2: -500.6704102, 2148.4094238, -520.5569458, 2234.4389648, -2735.1093750, 2668.9660645
3: -1078.5261230, 1921.1632080, -1120.6684570, 1999.0828857, -3077.6088867, 3041.8312988
4: -863.2470093, 2009.3989258, -897.9285889, 2090.1264648, -2953.3732910, 2907.3276367

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7373326, upper bound: 1781.7373326
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7373326, upper bound: 1781.7375634
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -432.3953552, 1746.4383545, -433.3481445, 1752.6160889, -2185.0114746, 2179.7866211
1: -695.5231323, 1934.6020508, -696.6046753, 1940.9016113, -2636.4243164, 2631.2062988
2: -519.4005127, 2226.5065918, -520.4738770, 2234.0847168, -2753.4848633, 2746.9804688
3: -1119.0119629, 1992.5078125, -1120.4903564, 1998.7620850, -3117.7736816, 3112.9980469
4: -895.4044189, 2083.7121582, -897.7866211, 2089.7934570, -2985.1977539, 2981.4980469

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7375655, upper bound: 1781.7374782
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7375655, upper bound: 1781.7377111
time: 0.62 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.66 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7349742, upper bound: 1781.7349742
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7349742, upper bound: 1781.7368072
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7368072, upper bound: 1781.7360732
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7368072, upper bound: 1781.7379062
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7350761, upper bound: 1781.7348478
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7352790, upper bound: 1781.7352294
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7368925, upper bound: 1781.7359469
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7370954, upper bound: 1781.7363285
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7348478, upper bound: 1781.7350761
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7359469, upper bound: 1781.7368925
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7352294, upper bound: 1781.7352790
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7363285, upper bound: 1781.7370954
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7388464, upper bound: 1781.7388464
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7388464, upper bound: 1781.7391742
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7388464, upper bound: 1781.7388635
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7388464, upper bound: 1781.7393369
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7350516, upper bound: 1781.7330078
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7350516, upper bound: 1781.7364283
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7362813, upper bound: 1781.7340719
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7362813, upper bound: 1781.7374924
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7342837, upper bound: 1781.7334345
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7344867, upper bound: 1781.7335990
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7354294, upper bound: 1781.7344986
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7356323, upper bound: 1781.7346632
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7364117, upper bound: 1781.7351532
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7364117, upper bound: 1781.7370936
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7366203, upper bound: 1781.7352988
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7366203, upper bound: 1781.7371761
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7373326, upper bound: 1781.7373326
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7373326, upper bound: 1781.7375655
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7377178, upper bound: 1781.7374782
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7377178, upper bound: 1781.7377111
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7330078, upper bound: 1781.7350516
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7340719, upper bound: 1781.7362813
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7330078, upper bound: 1781.7361352
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7340719, upper bound: 1781.7373649
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7330078, upper bound: 1781.7370836
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7340719, upper bound: 1781.7375275
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7330078, upper bound: 1781.7375445
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7340719, upper bound: 1781.7383981
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7334345, upper bound: 1781.7342837
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7344986, upper bound: 1781.7354294
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7335990, upper bound: 1781.7344867
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7346632, upper bound: 1781.7356323
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7373326, upper bound: 1781.7373326
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7373326, upper bound: 1781.7377178
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7375655, upper bound: 1781.7374782
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7375655, upper bound: 1781.7378634
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7350667, upper bound: 1781.7361767
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7351796, upper bound: 1781.7359961
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7370071, upper bound: 1781.7373007
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7371190, upper bound: 1781.7371201
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7351532, upper bound: 1781.7364117
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7352988, upper bound: 1781.7364679
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7370936, upper bound: 1781.7375356
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7372393, upper bound: 1781.7375919
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7360941, upper bound: 1781.7350575
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7372562, upper bound: 1781.7370064
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7359471, upper bound: 1781.7351668
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7371097, upper bound: 1781.7371207
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7373326, upper bound: 1781.7373326
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7373326, upper bound: 1781.7375634
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7375655, upper bound: 1781.7374782
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -1781.7375655, upper bound: 1781.7377111

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -242.2535858, 982.2024536, -242.2535858, 982.2024536, -1224.4560547, 1224.4560547
1: -390.3833923, 1087.1320801, -390.3833923, 1087.1320801, -1477.5155029, 1477.5155029
2: -290.8264771, 1252.6706543, -290.8264771, 1252.6706543, -1543.4970703, 1543.4970703
3: -627.6691895, 1117.2794189, -627.6691895, 1117.2794189, -1744.9486084, 1744.9486084
4: -500.9706116, 1171.2468262, -500.9706116, 1171.2468262, -1672.2174072, 1672.2174072

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7344303, upper bound: 1781.7342827
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7342082, upper bound: 1781.7341707
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -242.2535858, 982.2024536, -253.0717621, 1026.6875000, -1268.9410400, 1235.2741699
1: -390.3833923, 1087.1320801, -407.7406311, 1136.2047119, -1526.5881348, 1494.8726807
2: -290.8264771, 1252.6706543, -303.8264465, 1309.1923828, -1600.0187988, 1556.4970703
3: -627.6691895, 1117.2794189, -655.2677612, 1168.8730469, -1796.5422363, 1772.5471191
4: -500.9706116, 1171.2468262, -523.6318359, 1224.8913574, -1725.8619385, 1694.8786621

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7344303, upper bound: 1781.7343493
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7342082, upper bound: 1781.7342373
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -253.0717621, 1026.6875000, -242.2535858, 982.2024536, -1235.2740479, 1268.9410400
1: -407.7406311, 1136.2047119, -390.3833923, 1087.1320801, -1494.8726807, 1526.5881348
2: -303.8264465, 1309.1923828, -290.8264771, 1252.6706543, -1556.4970703, 1600.0187988
3: -655.2677612, 1168.8730469, -627.6691895, 1117.2794189, -1772.5471191, 1796.5422363
4: -523.6318359, 1224.8913574, -500.9706116, 1171.2468262, -1694.8786621, 1725.8619385

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7362753, upper bound: 1781.7353818
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7342329, upper bound: 1781.7343037
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -253.0717621, 1026.6875000, -253.0717621, 1026.6875000, -1279.7591553, 1279.7590332
1: -407.7406311, 1136.2047119, -407.7406311, 1136.2047119, -1543.9453125, 1543.9453125
2: -303.8264465, 1309.1923828, -303.8264465, 1309.1923828, -1613.0187988, 1613.0187988
3: -655.2677612, 1168.8730469, -655.2677612, 1168.8730469, -1824.1408691, 1824.1408691
4: -523.6318359, 1224.8913574, -523.6318359, 1224.8913574, -1748.5231934, 1748.5231934

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7362754, upper bound: 1781.7354484
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7342329, upper bound: 1781.7343703
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -242.2535858, 982.2024536, -271.3430481, 1101.4902344, -1343.7437744, 1253.5452881
1: -390.3833923, 1087.1320801, -437.6978760, 1219.0781250, -1609.4614258, 1524.8299561
2: -290.8264771, 1252.6706543, -326.5375061, 1403.8286133, -1694.6550293, 1579.2081299
3: -627.6691895, 1117.2794189, -702.6085205, 1254.8631592, -1882.5323486, 1819.8879395
4: -500.9706116, 1171.2468262, -561.8131104, 1314.3068848, -1815.2774658, 1733.0599365

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7345882, upper bound: 1781.7336945
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7303516, upper bound: 1781.7313077
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7348771, upper bound: 1781.7347573
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -242.1595917, 981.8240967, -289.7269592, 1171.9594727, -1414.1187744, 1271.5510254
1: -390.2310791, 1086.7126465, -467.4779053, 1297.0390625, -1687.2701416, 1554.1903076
2: -290.7125549, 1252.1887207, -348.6601868, 1494.1398926, -1784.8522949, 1600.8488770
3: -627.4274902, 1116.8399658, -749.8610229, 1336.6997070, -1964.1271973, 1866.7008057
4: -500.7748108, 1170.7946777, -599.4759521, 1399.8709717, -1900.6456299, 1770.2705078

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7329152, upper bound: 1781.7322545
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7328593, upper bound: 1781.7321424
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -253.0717621, 1026.6875000, -272.0064392, 1104.1983643, -1357.2701416, 1298.6937256
1: -407.7406311, 1136.2047119, -438.7653198, 1222.0687256, -1629.8093262, 1574.9699707
2: -303.8264465, 1309.1923828, -327.3316650, 1407.2741699, -1711.1005859, 1636.5239258
3: -655.2677612, 1168.8730469, -704.3140259, 1257.9733887, -1913.2412109, 1873.1870117
4: -523.6318359, 1224.8913574, -563.1935425, 1317.5512695, -1841.1831055, 1788.0849609

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7366935, upper bound: 1781.7357480
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7364142, upper bound: 1781.7356904
time: 0.51 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -252.9728088, 1026.2916260, -290.3930359, 1174.6596680, -1427.6324463, 1316.6844482
1: -407.5806274, 1135.7652588, -468.5578918, 1300.0351562, -1707.6157227, 1604.3231201
2: -303.7065125, 1308.6877441, -349.4653015, 1497.5777588, -1801.2840576, 1658.1530762
3: -655.0134277, 1168.4130859, -751.5907593, 1339.8083496, -1994.8217773, 1920.0037842
4: -523.4254761, 1224.4166260, -600.8707275, 1403.1135254, -1926.5389404, 1825.2873535

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7346699, upper bound: 1781.7333535
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7335194, upper bound: 1781.7322755
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -271.3430481, 1101.4902344, -242.2535858, 982.2024536, -1253.5452881, 1343.7437744
1: -437.6978760, 1219.0781250, -390.3833923, 1087.1320801, -1524.8299561, 1609.4614258
2: -326.5375061, 1403.8286133, -290.8264771, 1252.6706543, -1579.2081299, 1694.6550293
3: -702.6085205, 1254.8631592, -627.6691895, 1117.2794189, -1819.8879395, 1882.5323486
4: -561.8131104, 1314.3068848, -500.9706116, 1171.2468262, -1733.0599365, 1815.2774658

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7341640, upper bound: 1781.7344012
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7341640, upper bound: 1781.7350761
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -272.0064392, 1104.1983643, -253.0717621, 1026.6875000, -1298.6937256, 1357.2701416
1: -438.7653198, 1222.0687256, -407.7406311, 1136.2047119, -1574.9699707, 1629.8093262
2: -327.3316650, 1407.2741699, -303.8264465, 1309.1923828, -1636.5239258, 1711.1005859
3: -704.3140259, 1257.9733887, -655.2677612, 1168.8730469, -1873.1870117, 1913.2412109
4: -563.1935425, 1317.5512695, -523.6318359, 1224.8913574, -1788.0849609, 1841.1831055

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7352630, upper bound: 1781.7362176
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7352630, upper bound: 1781.7368925
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -289.7269592, 1171.9594727, -242.1595917, 981.8240967, -1271.5510254, 1414.1187744
1: -467.4779053, 1297.0390625, -390.2310791, 1086.7126465, -1554.1904297, 1687.2701416
2: -348.6601868, 1494.1398926, -290.7125549, 1252.1887207, -1600.8488770, 1784.8522949
3: -749.8610229, 1336.6997070, -627.4274902, 1116.8399658, -1866.7008057, 1964.1271973
4: -599.4759521, 1399.8709717, -500.7748108, 1170.7946777, -1770.2705078, 1900.6456299

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7340565, upper bound: 1781.7345148
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7340565, upper bound: 1781.7352790
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -290.3930359, 1174.6596680, -252.9728088, 1026.2916260, -1316.6844482, 1427.6324463
1: -468.5578918, 1300.0351562, -407.5806274, 1135.7652588, -1604.3231201, 1707.6157227
2: -349.4653015, 1497.5777588, -303.7065125, 1308.6877441, -1658.1530762, 1801.2840576
3: -751.5907593, 1339.8083496, -655.0134277, 1168.4130859, -1920.0037842, 1994.8217773
4: -600.8707275, 1403.1135254, -523.4254761, 1224.4166260, -1825.2873535, 1926.5389404

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7343264, upper bound: 1781.7363312
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7351555, upper bound: 1781.7370954
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -272.0064392, 1104.1983643, -272.0064392, 1104.1983643, -1376.2048340, 1376.2048340
1: -438.7653198, 1222.0687256, -438.7653198, 1222.0687256, -1660.8339844, 1660.8339844
2: -327.3316650, 1407.2741699, -327.3316650, 1407.2741699, -1734.6057129, 1734.6057129
3: -704.3140259, 1257.9733887, -704.3140259, 1257.9733887, -1962.2873535, 1962.2873535
4: -563.1935425, 1317.5512695, -563.1935425, 1317.5512695, -1880.7448730, 1880.7448730

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7360127, upper bound: 1781.7359391
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7366966, upper bound: 1781.7366141
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -272.0064392, 1104.1983643, -290.3930359, 1174.6596680, -1446.6661377, 1394.5911865
1: -438.7653198, 1222.0687256, -468.5578918, 1300.0351562, -1738.8005371, 1690.6265869
2: -327.3316650, 1407.2741699, -349.4653015, 1497.5777588, -1824.9093018, 1756.7395020
3: -704.3140259, 1257.9733887, -751.5907593, 1339.8083496, -2044.1223145, 2009.5639648
4: -563.1935425, 1317.5512695, -600.8707275, 1403.1135254, -1966.3071289, 1918.4219971

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7360127, upper bound: 1781.7364032
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7366966, upper bound: 1781.7370782
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -290.3930359, 1174.6596680, -272.0064392, 1104.1983643, -1394.5913086, 1446.6661377
1: -468.5578918, 1300.0351562, -438.7653198, 1222.0687256, -1690.6265869, 1738.8005371
2: -349.4653015, 1497.5777588, -327.3316650, 1407.2741699, -1756.7395020, 1824.9093018
3: -751.5907593, 1339.8083496, -704.3140259, 1257.9733887, -2009.5639648, 2044.1223145
4: -600.8707275, 1403.1135254, -563.1935425, 1317.5512695, -1918.4219971, 1966.3071289

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7358603, upper bound: 1781.7360527
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7366141, upper bound: 1781.7366254
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -290.3930359, 1174.6596680, -290.3930359, 1174.6596680, -1465.0526123, 1465.0526123
1: -468.5578918, 1300.0351562, -468.5578918, 1300.0351562, -1768.5930176, 1768.5930176
2: -349.4653015, 1497.5777588, -349.4653015, 1497.5777588, -1847.0430908, 1847.0430908
3: -751.5907593, 1339.8083496, -751.5907593, 1339.8083496, -2091.3991699, 2091.3991699
4: -600.8707275, 1403.1135254, -600.8707275, 1403.1135254, -2003.9842529, 2003.9842529

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7358603, upper bound: 1781.7364473
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7366141, upper bound: 1781.7371637
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -242.2535858, 982.2024536, -339.9133301, 1373.2429199, -1615.4964600, 1322.1156006
1: -390.3833923, 1087.1320801, -547.4383545, 1520.6569824, -1911.0404053, 1634.5704346
2: -290.8264771, 1252.6706543, -408.4419250, 1751.1427002, -2041.9691162, 1661.1125488
3: -627.6691895, 1117.2794189, -879.7506714, 1566.9632568, -2194.6320801, 1997.0300293
4: -500.9706116, 1171.2468262, -704.6979980, 1638.8487549, -2139.8193359, 1875.9445801

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7336653, upper bound: 1781.7281531
time: 0.50 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7338976, upper bound: 1781.7320186
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7338203, upper bound: 1781.7323128
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7350516, upper bound: 1781.7330078
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -242.2535858, 982.2024536, -373.3807068, 1509.6898193, -1751.9433594, 1355.5831299
1: -390.3833923, 1087.1320801, -601.2086792, 1671.9190674, -2062.3022461, 1688.3408203
2: -290.8264771, 1252.6706543, -449.2063599, 1924.1254883, -2214.9519043, 1701.8769531
3: -627.6691895, 1117.2794189, -965.2205200, 1724.4213867, -2352.0903320, 2082.5000000
4: -500.9706116, 1171.2468262, -774.8056030, 1801.6715088, -2302.6416016, 1946.0524902

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7336653, upper bound: 1781.7343038
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7338976, upper bound: 1781.7344691
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7338203, upper bound: 1781.7357333
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7350516, upper bound: 1781.7364283
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -253.0717621, 1026.6875000, -340.4019470, 1375.2512207, -1628.3228760, 1367.0891113
1: -407.7406311, 1136.2047119, -548.2256470, 1522.8673096, -1930.6079102, 1684.4304199
2: -303.8264465, 1309.1923828, -409.0312500, 1753.7003174, -2057.5266113, 1718.2236328
3: -655.2677612, 1168.8730469, -881.0024414, 1569.2692871, -2224.5371094, 2049.8754883
4: -523.6318359, 1224.8913574, -705.7205811, 1641.2520752, -2164.8837891, 1930.6119385

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7347670, upper bound: 1781.7292172
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7348190, upper bound: 1781.7331276
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7262824, upper bound: 1781.7208606
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7074802, upper bound: 1781.7130106
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -253.0717621, 1026.6875000, -373.8887634, 1511.7923584, -1764.8641357, 1400.5761719
1: -407.7406311, 1136.2047119, -602.0303955, 1674.2281494, -2081.9685059, 1738.2349854
2: -303.8264465, 1309.1923828, -449.8215332, 1926.8049316, -2230.6311035, 1759.0139160
3: -655.2677612, 1168.8730469, -966.5252686, 1726.8314209, -2382.0991211, 2135.3981934
4: -523.6318359, 1224.8913574, -775.8743896, 1804.1857910, -2327.8173828, 2000.7655029

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7347670, upper bound: 1781.7353679
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7348190, upper bound: 1781.7355780
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7262824, upper bound: 1781.7243629
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7074802, upper bound: 1781.7165128
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -242.0127258, 981.2075806, -414.3564148, 1675.3409424, -1917.3535156, 1395.5638428
1: -389.9958191, 1086.0404053, -666.1355591, 1855.7004395, -2245.6958008, 1752.1760254
2: -290.5358582, 1251.4107666, -497.6574402, 2135.6311035, -2426.1667480, 1749.0682373
3: -627.0516968, 1116.1368408, -1072.1060791, 1909.7611084, -2536.8127441, 2188.2424316
4: -500.4738464, 1170.0587158, -858.1303101, 1997.4503174, -2497.9238281, 2028.1888428

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7339215, upper bound: 1781.7321835
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7300913, upper bound: 1781.7298944
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7340848, upper bound: 1781.7333439
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -241.9187927, 980.8300781, -429.7855225, 1735.8510742, -1977.7698975, 1410.6156006
1: -389.8438721, 1085.6214600, -691.2695312, 1922.9487305, -2312.7924805, 1776.8906250
2: -290.4222107, 1250.9296875, -516.2390137, 2213.0256348, -2503.4475098, 1767.1687012
3: -626.8104858, 1115.6983643, -1112.2221680, 1980.5042725, -2607.3146973, 2227.9201660
4: -500.2783203, 1169.6071777, -890.0374756, 2071.0859375, -2571.3642578, 2059.6445312

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7340304, upper bound: 1781.7319272
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7303087, upper bound: 1781.7300212
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7343021, upper bound: 1781.7334708
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -252.8373108, 1025.7237549, -414.8493958, 1677.3507080, -1930.1879883, 1440.5731201
1: -407.3605347, 1135.1484375, -666.9367065, 1857.9216309, -2265.2819824, 1802.0852051
2: -303.5417480, 1307.9656982, -498.2525940, 2138.1972656, -2441.7387695, 1806.2182617
3: -654.6620483, 1167.7681885, -1073.3891602, 1912.0555420, -2566.7172852, 2241.1572266
4: -523.1476440, 1223.7237549, -859.1561890, 1999.8604736, -2523.0080566, 2082.8798828

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7351888, upper bound: 1781.7342559
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7352304, upper bound: 1781.7342634
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -252.7385406, 1025.3287354, -430.3157654, 1737.9997559, -1990.7382812, 1455.6445312
1: -407.2008972, 1134.7102051, -692.1263428, 1925.3264160, -2332.5273438, 1826.8364258
2: -303.4219360, 1307.4621582, -516.8775024, 2215.7617188, -2519.1835938, 1824.3392334
3: -654.4080200, 1167.3092041, -1113.5946045, 1982.9691162, -2637.3771973, 2280.9038086
4: -522.9416504, 1223.2498779, -891.1426392, 2073.6633301, -2596.6044922, 2114.3923340

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7354062, upper bound: 1781.7343828
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7354478, upper bound: 1781.7343903
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -272.0064392, 1104.1983643, -340.4019470, 1375.2512207, -1647.2576904, 1444.6002197
1: -438.7653198, 1222.0687256, -548.2256470, 1522.8673096, -1961.6324463, 1770.2944336
2: -327.3316650, 1407.2741699, -409.0312500, 1753.7003174, -2081.0317383, 1816.3054199
3: -704.3140259, 1257.9733887, -881.0024414, 1569.2692871, -2273.5832520, 2138.9758301
4: -563.1935425, 1317.5512695, -705.7205811, 1641.2520752, -2204.4455566, 2023.2717285

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7354836, upper bound: 1781.7309312
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7340220, upper bound: 1781.7327990
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7342910, upper bound: 1781.7334739
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -272.0064392, 1104.1983643, -373.8887634, 1511.7923584, -1783.7988281, 1478.0871582
1: -438.7653198, 1222.0687256, -602.0303955, 1674.2281494, -2112.9929199, 1824.0988770
2: -327.3316650, 1407.2741699, -449.8215332, 1926.8049316, -2254.1364746, 1857.0957031
3: -704.3140259, 1257.9733887, -966.5252686, 1726.8314209, -2431.1455078, 2224.4985352
4: -563.1935425, 1317.5512695, -775.8743896, 1804.1857910, -2367.3789062, 2093.4250488

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7354836, upper bound: 1781.7342318
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7340220, upper bound: 1781.7342731
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7342910, upper bound: 1781.7349481
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -290.3930359, 1174.6596680, -340.3200989, 1374.9171143, -1665.3099365, 1514.9797363
1: -468.5578918, 1300.0351562, -548.0930786, 1522.4979248, -1991.0556641, 1848.1279297
2: -349.4653015, 1497.5777588, -408.9316711, 1753.2745361, -2102.7397461, 1906.5093994
3: -751.5907593, 1339.8083496, -880.7910156, 1568.8850098, -2320.4755859, 2220.5993652
4: -600.8707275, 1403.1135254, -705.5502319, 1640.8537598, -2241.7246094, 2108.6633301

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7352860, upper bound: 1781.7310768
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7340822, upper bound: 1781.7329126
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7347442, upper bound: 1781.7336769
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -290.3930359, 1174.6596680, -373.8252563, 1511.5307617, -1801.9234619, 1548.4848633
1: -468.5578918, 1300.0351562, -601.9265137, 1673.9389648, -2142.4965820, 1901.9616699
2: -349.4653015, 1497.5777588, -449.7445374, 1926.4721680, -2275.9375000, 1947.3222656
3: -751.5907593, 1339.8083496, -966.3622437, 1726.5311279, -2478.1215820, 2306.1704102
4: -600.8707275, 1403.1135254, -775.7431030, 1803.8747559, -2404.7451172, 2178.8566895

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7352860, upper bound: 1781.7343174
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7340822, upper bound: 1781.7343867
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7347442, upper bound: 1781.7350965
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -271.8345642, 1103.4819336, -416.8405151, 1685.3878174, -1957.2222900, 1520.3221436
1: -438.4944763, 1221.2904053, -670.1839600, 1866.7962646, -2305.2907715, 1891.4743652
2: -327.1272278, 1406.3658447, -500.6704102, 2148.4094238, -2475.5366211, 1907.0361328
3: -703.8865967, 1257.1531982, -1078.5261230, 1921.1632080, -2625.0483398, 2335.6791992
4: -562.8423462, 1316.6882324, -863.2470093, 2009.3989258, -2572.2412109, 2179.9350586

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7350142, upper bound: 1781.7345271
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7352832, upper bound: 1781.7352020
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -271.8345642, 1103.4819336, -432.3953552, 1746.4383545, -2018.2727051, 1535.8770752
1: -438.4944763, 1221.2904053, -695.5231323, 1934.6020508, -2373.0961914, 1916.8134766
2: -327.1272278, 1406.3658447, -519.4005127, 2226.5065918, -2553.6337891, 1925.7662354
3: -703.8865967, 1257.1531982, -1119.0119629, 1992.5078125, -2696.3940430, 2376.1650391
4: -562.8423462, 1316.6882324, -895.4044189, 2083.7121582, -2646.5534668, 2212.0925293

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7350142, upper bound: 1781.7347729
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7352832, upper bound: 1781.7354478
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -290.1746826, 1173.7753906, -416.8405151, 1685.3878174, -1975.5625000, 1590.6158447
1: -468.2087097, 1299.0659180, -670.1839600, 1866.7962646, -2335.0048828, 1969.2498779
2: -349.2021790, 1496.4548340, -500.6704102, 2148.4094238, -2497.6115723, 1997.1250000
3: -751.0340576, 1338.7947998, -1078.5261230, 1921.1632080, -2672.1955566, 2417.3208008
4: -600.4206543, 1402.0515137, -863.2470093, 2009.3989258, -2609.8195801, 2265.2980957

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7350744, upper bound: 1781.7346407
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7357365, upper bound: 1781.7354049
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -290.1746826, 1173.7753906, -432.3953552, 1746.4383545, -2036.6130371, 1606.1707764
1: -468.2087097, 1299.0659180, -695.5231323, 1934.6020508, -2402.8105469, 1994.5891113
2: -349.2021790, 1496.4548340, -519.4005127, 2226.5065918, -2575.7087402, 2015.8551025
3: -751.0340576, 1338.7947998, -1119.0119629, 1992.5078125, -2743.5412598, 2457.8063965
4: -600.4206543, 1402.0515137, -895.4044189, 2083.7121582, -2684.1320801, 2297.4558105

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7350744, upper bound: 1781.7348660
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7357365, upper bound: 1781.7356501
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -339.9133301, 1373.2429199, -242.2535858, 982.2024536, -1322.1156006, 1615.4964600
1: -547.4383545, 1520.6569824, -390.3833923, 1087.1320801, -1634.5704346, 1911.0404053
2: -408.4419250, 1751.1427002, -290.8264771, 1252.6706543, -1661.1125488, 2041.9691162
3: -879.7506714, 1566.9632568, -627.6691895, 1117.2794189, -1997.0300293, 2194.6320801
4: -704.6979980, 1638.8487549, -500.9706116, 1171.2468262, -1875.9445801, 2139.8193359

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7326629, upper bound: 1781.7346402
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7325466, upper bound: 1781.7346348
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7287270, upper bound: 1781.7332075
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -340.4019470, 1375.2512207, -253.0717621, 1026.6875000, -1367.0891113, 1628.3229980
1: -548.2256470, 1522.8673096, -407.7406311, 1136.2047119, -1684.4304199, 1930.6079102
2: -409.0312500, 1753.7003174, -303.8264465, 1309.1923828, -1718.2236328, 2057.5266113
3: -881.0024414, 1569.2692871, -655.2677612, 1168.8730469, -2049.8754883, 2224.5371094
4: -705.7205811, 1641.2520752, -523.6318359, 1224.8913574, -1930.6119385, 2164.8837891

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7240878, upper bound: 1781.7135644
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7130106, upper bound: 1781.7077639
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -373.3807068, 1509.6898193, -242.2535858, 982.2024536, -1355.5831299, 1751.9433594
1: -601.2086792, 1671.9190674, -390.3833923, 1087.1320801, -1688.3408203, 2062.3022461
2: -449.2063599, 1924.1254883, -290.8264771, 1252.6706543, -1701.8769531, 2214.9519043
3: -965.2205200, 1724.4213867, -627.6691895, 1117.2794189, -2082.5000000, 2352.0903320
4: -774.8056030, 1801.6715088, -500.9706116, 1171.2468262, -1946.0524902, 2302.6411133

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7332086, upper bound: 1781.7342146
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7330993, upper bound: 1781.7343750
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -373.8887634, 1511.7923584, -253.0717621, 1026.6875000, -1400.5761719, 1764.8640137
1: -602.0303955, 1674.2281494, -407.7406311, 1136.2047119, -1738.2348633, 2081.9685059
2: -449.8215332, 1926.8049316, -303.8264465, 1309.1923828, -1759.0139160, 2230.6311035
3: -966.5252686, 1726.8314209, -655.2677612, 1168.8730469, -2135.3981934, 2382.0991211
4: -775.8743896, 1804.1857910, -523.6318359, 1224.8913574, -2000.7655029, 2327.8173828

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7342727, upper bound: 1781.7353602
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7341634, upper bound: 1781.7355207
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -339.9133301, 1373.2429199, -274.6368103, 1114.2810059, -1454.1942139, 1647.8796387
1: -547.4383545, 1520.6569824, -442.6131592, 1233.6594238, -1781.0977783, 1963.2701416
2: -408.4419250, 1751.1427002, -330.3896484, 1420.2746582, -1828.7165527, 2081.5314941
3: -879.7506714, 1566.9632568, -710.6918335, 1269.7735596, -2149.5241699, 2277.6550293
4: -704.6979980, 1638.8487549, -568.9721069, 1328.8990479, -2033.5966797, 2207.8208008

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7338890, upper bound: 1781.7366722
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7325466, upper bound: 1781.7348364
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7287270, upper bound: 1781.7334091
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -340.4019470, 1375.2512207, -284.2988892, 1154.0219727, -1494.4238281, 1659.5500488
1: -548.2256470, 1522.8673096, -458.2223511, 1277.4332275, -1825.6589355, 1981.0894775
2: -409.0312500, 1753.7003174, -341.9913635, 1470.7464600, -1879.7775879, 2095.6916504
3: -881.0024414, 1569.2692871, -735.3759155, 1315.6445312, -2196.6464844, 2304.6447754
4: -705.7205811, 1641.2520752, -589.1831055, 1376.7503662, -2082.4709473, 2230.4348145

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7346206, upper bound: 1781.7371161
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7323465, upper bound: 1781.7339912
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7285269, upper bound: 1781.7325640
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -373.3807068, 1509.6898193, -274.6368103, 1114.2810059, -1487.6617432, 1784.3266602
1: -601.2086792, 1671.9190674, -442.6131592, 1233.6594238, -1834.8681641, 2114.5322266
2: -449.2063599, 1924.1254883, -330.3896484, 1420.2746582, -1869.4809570, 2254.5146484
3: -965.2205200, 1724.4213867, -710.6918335, 1269.7735596, -2234.9938965, 2435.1132812
4: -774.8056030, 1801.6715088, -568.9721069, 1328.8990479, -2103.7045898, 2370.6430664

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7344960, upper bound: 1781.7350053
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7343760, upper bound: 1781.7351548
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -373.8887634, 1511.7923584, -284.2988892, 1154.0219727, -1527.9107666, 1796.0913086
1: -602.0303955, 1674.2281494, -458.2223511, 1277.4332275, -1879.4633789, 2132.4499512
2: -449.8215332, 1926.8049316, -341.9913635, 1470.7464600, -1920.5678711, 2268.7963867
3: -966.5252686, 1726.8314209, -735.3759155, 1315.6445312, -2282.1689453, 2462.2070312
4: -775.8743896, 1804.1857910, -589.1831055, 1376.7503662, -2152.6245117, 2393.3684082

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7352603, upper bound: 1781.7356673
time: 0.51 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7351510, upper bound: 1781.7358278
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -414.3564148, 1675.3409424, -242.0127258, 981.2075806, -1395.5638428, 1917.3535156
1: -666.1355591, 1855.7004395, -389.9958191, 1086.0404053, -1752.1760254, 2245.6958008
2: -497.6574402, 2135.6311035, -290.5358582, 1251.4107666, -1749.0682373, 2426.1667480
3: -1072.1060791, 1909.7611084, -627.0516968, 1116.1368408, -2188.2426758, 2536.8127441
4: -858.1303101, 1997.4503174, -500.4738464, 1170.0587158, -2028.1888428, 2497.9238281

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7333224, upper bound: 1781.7342347
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7334345, upper bound: 1781.7322697
time: 0.50 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7334345, upper bound: 1781.7338747
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -414.8493958, 1677.3507080, -252.8373108, 1025.7237549, -1440.5731201, 1930.1879883
1: -666.9367065, 1857.9216309, -407.3605347, 1135.1484375, -1802.0852051, 2265.2819824
2: -498.2525940, 2138.1972656, -303.5417480, 1307.9656982, -1806.2182617, 2441.7387695
3: -1073.3891602, 1912.0555420, -654.6620483, 1167.7681885, -2241.1572266, 2566.7172852
4: -859.1561890, 1999.8604736, -523.1476440, 1223.7237549, -2082.8798828, 2523.0080566

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7248416, upper bound: 1781.7144829
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7334555, upper bound: 1781.7345158
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7344986, upper bound: 1781.7334154
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7344986, upper bound: 1781.7350204
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -429.7855225, 1735.8510742, -241.9187927, 980.8300781, -1410.6156006, 1977.7698975
1: -691.2695312, 1922.9487305, -389.8438721, 1085.6214600, -1776.8906250, 2312.7924805
2: -516.2390137, 2213.0256348, -290.4222107, 1250.9296875, -1767.1687012, 2503.4475098
3: -1112.2221680, 1980.5042725, -626.8104858, 1115.6983643, -2227.9201660, 2607.3146973
4: -890.0374756, 2071.0859375, -500.2783203, 1169.6071777, -2059.6445312, 2571.3642578

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7324596, upper bound: 1781.7334541
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7324596, upper bound: 1781.7344867
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -430.3157654, 1737.9997559, -252.7385406, 1025.3287354, -1455.6444092, 1990.7382812
1: -692.1263428, 1925.3264160, -407.2008972, 1134.7102051, -1826.8364258, 2332.5273438
2: -516.8775024, 2215.7617188, -303.4219360, 1307.4621582, -1824.3392334, 2519.1835938
3: -1113.5946045, 1982.9691162, -654.4080200, 1167.3092041, -2280.9038086, 2637.3771973
4: -891.1426392, 2073.6633301, -522.9416504, 1223.2498779, -2114.3923340, 2596.6044922

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7335237, upper bound: 1781.7345997
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7335237, upper bound: 1781.7356323
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -416.8405151, 1685.3878174, -271.8345642, 1103.4819336, -1520.3221436, 1957.2224121
1: -670.1839600, 1866.7962646, -438.4944763, 1221.2904053, -1891.4743652, 2305.2907715
2: -500.6704102, 2148.4094238, -327.1272278, 1406.3658447, -1907.0361328, 2475.5366211
3: -1078.5261230, 1921.1632080, -703.8865967, 1257.1531982, -2335.6791992, 2625.0483398
4: -863.2470093, 2009.3989258, -562.8423462, 1316.6882324, -2179.9350586, 2572.2412109

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7360298, upper bound: 1781.7328507
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7370599, upper bound: 1781.7368833
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -416.8405151, 1685.3878174, -290.1746826, 1173.7753906, -1590.6158447, 1975.5625000
1: -670.1839600, 1866.7962646, -468.2087097, 1299.0659180, -1969.2498779, 2335.0048828
2: -500.6704102, 2148.4094238, -349.2021790, 1496.4548340, -1997.1250000, 2497.6115723
3: -1078.5261230, 1921.1632080, -751.0340576, 1338.7947998, -2417.3205566, 2672.1955566
4: -863.2470093, 2009.3989258, -600.4206543, 1402.0515137, -2265.2983398, 2609.8195801

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7360298, upper bound: 1781.7331789
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7370599, upper bound: 1781.7372115
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -432.3953552, 1746.4383545, -271.8345642, 1103.4819336, -1535.8770752, 2018.2727051
1: -695.5231323, 1934.6020508, -438.4944763, 1221.2904053, -1916.8134766, 2373.0961914
2: -519.4005127, 2226.5065918, -327.1272278, 1406.3658447, -1925.7662354, 2553.6337891
3: -1119.0119629, 1992.5078125, -703.8865967, 1257.1531982, -2376.1650391, 2696.3940430
4: -895.4044189, 2083.7121582, -562.8423462, 1316.6882324, -2212.0925293, 2646.5534668

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7368450, upper bound: 1781.7370003
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7373028, upper bound: 1781.7372805
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -432.3953552, 1746.4383545, -290.1746826, 1173.7753906, -1606.1707764, 2036.6130371
1: -695.5231323, 1934.6020508, -468.2087097, 1299.0659180, -1994.5891113, 2402.8105469
2: -519.4005127, 2226.5065918, -349.2021790, 1496.4548340, -2015.8551025, 2575.7087402
3: -1119.0119629, 1992.5078125, -751.0340576, 1338.7947998, -2457.8063965, 2743.5412598
4: -895.4044189, 2083.7121582, -600.4206543, 1402.0515137, -2297.4555664, 2684.1318359

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7368450, upper bound: 1781.7373614
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7373028, upper bound: 1781.7375691
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -340.4019470, 1375.2512207, -363.1659851, 1468.0748291, -1808.4766846, 1738.4172363
1: -548.2256470, 1522.8673096, -584.7805786, 1625.7316895, -2173.9572754, 2107.6479492
2: -409.0312500, 1753.7003174, -436.7915039, 1871.4442139, -2280.4753418, 2190.4916992
3: -881.0024414, 1569.2692871, -939.8736572, 1675.1641846, -2556.1665039, 2509.1430664
4: -705.7205811, 1641.2520752, -752.9602051, 1751.5958252, -2457.3164062, 2394.2124023

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7303079, upper bound: 1781.7339396
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7302015, upper bound: 1781.7338900
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -340.3200989, 1374.9171143, -376.2657166, 1519.8453369, -1860.1652832, 1751.1826172
1: -548.0930786, 1522.4979248, -606.2966919, 1683.0556641, -2231.1484375, 2128.7944336
2: -408.9316711, 1753.2745361, -452.6686096, 1937.5261230, -2346.4575195, 2205.9431152
3: -880.7910156, 1568.8850098, -973.7809448, 1735.8786621, -2616.6696777, 2542.6655273
4: -705.5502319, 1640.8537598, -780.1737061, 1814.7879639, -2520.3378906, 2421.0273438

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7303313, upper bound: 1781.7323118
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7302368, upper bound: 1781.7322831
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -373.8887634, 1511.7923584, -365.1392517, 1476.0261230, -1849.9149170, 1876.9316406
1: -602.0303955, 1674.2281494, -588.0028687, 1634.5186768, -2236.5490723, 2262.2309570
2: -449.8215332, 1926.8049316, -439.1896667, 1881.5516357, -2331.3730469, 2365.9946289
3: -966.5252686, 1726.8314209, -944.9691162, 1684.2034912, -2650.7287598, 2671.8005371
4: -775.8743896, 1804.1857910, -757.0154419, 1761.0747070, -2536.9489746, 2561.2011719

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7369720, upper bound: 1781.7369720
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7369720, upper bound: 1781.7370850
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -373.8252563, 1511.5307617, -378.3398132, 1528.2536621, -1902.0787354, 1889.8704834
1: -601.9265137, 1673.9389648, -609.6896362, 1692.3015137, -2294.2280273, 2283.6284180
2: -449.7445374, 1926.4721680, -455.1920166, 1948.2338867, -2397.9782715, 2381.6640625
3: -966.3622437, 1726.5311279, -979.1234131, 1745.3703613, -2711.7326660, 2705.6545410
4: -775.7431030, 1803.8747559, -784.4274902, 1824.7851562, -2600.5283203, 2588.3012695

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7370837, upper bound: 1781.7370069
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7370837, upper bound: 1781.7371201
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -340.4019470, 1375.2512207, -414.8493958, 1677.3507080, -2017.7525635, 1790.1005859
1: -548.2256470, 1522.8673096, -666.9367065, 1857.9216309, -2406.1469727, 2189.8039551
2: -409.0312500, 1753.7003174, -498.2525940, 2138.1972656, -2547.2285156, 2251.9528809
3: -881.0024414, 1569.2692871, -1073.3891602, 1912.0555420, -2793.0578613, 2642.6584473
4: -705.7205811, 1641.2520752, -859.1561890, 1999.8604736, -2705.5810547, 2500.4082031

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=2073.053466796875
rel_dist={0: [-1781.7419244201583, 1781.7419244201583]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7403847, upper bound: 1781.7403847
time: 0.58 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7403847, upper bound: 1781.7403847
time: 0.59 seconds

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

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7393295, upper bound: 1781.7391794
time: 0.84 seconds

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

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7393295, upper bound: 1781.7391794
time: 0.76 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7394478, upper bound: 1781.7394478
time: 0.60 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.73 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.73
Output dim: 0, lower bound: -1781.7393295, upper bound: 1781.7391794
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.73
Output dim: 0, lower bound: -1781.7394478, upper bound: 1781.7394478
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.73
Output dim: 0, lower bound: -1781.7393295, upper bound: 1781.7391794
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.73
Output dim: 0, lower bound: -1781.7394478, upper bound: 1781.7394478

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -285.8988953, 1160.2062988, -374.1450806, 1512.7781982, -1798.6771240, 1534.3513184
1: -460.6930542, 1283.9232178, -602.2437744, 1674.7860107, -2135.4790039, 1886.1668701
2: -343.7970276, 1479.0144043, -449.8688354, 1928.4483643, -2272.2453613, 1928.8833008
3: -739.5973511, 1322.0988770, -967.2557373, 1726.6652832, -2466.2622070, 2289.3544922
4: -592.1936646, 1384.1098633, -775.8203125, 1805.0687256, -2397.2619629, 2159.9299316

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7383546, upper bound: 1781.7385103
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7393295, upper bound: 1781.7391794
time: 0.62 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -290.5298462, 1178.6455078, -427.8232422, 1729.8629150, -2020.3927002, 1606.4687500
1: -468.1905212, 1304.4074707, -687.5839844, 1915.6551514, -2383.8454590, 1991.9914551
2: -349.4006348, 1502.5856934, -513.7957153, 2205.1147461, -2554.5153809, 2016.3812256
3: -751.7043457, 1342.9464111, -1105.8681641, 1973.0588379, -2724.7631836, 2448.8144531
4: -601.8133545, 1406.0264893, -886.3011475, 2063.0727539, -2664.8859863, 2292.3276367

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7384967, upper bound: 1781.7387713
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7394478, upper bound: 1781.7394478
time: 0.65 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -399.2962341, 1614.5493164, -381.5436096, 1542.9018555, -1942.1981201, 1996.0928955
1: -642.5704346, 1787.4582520, -614.2192383, 1708.1495361, -2350.7194824, 2401.6772461
2: -479.9982910, 2057.8203125, -458.8776855, 1966.8120117, -2446.8103027, 2516.6977539
3: -1032.1456299, 1842.9288330, -986.3864746, 1761.2528076, -2793.3984375, 2829.3154297
4: -827.6176147, 1926.5274658, -791.3984375, 1840.9916992, -2668.6088867, 2717.9257812

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7390604, upper bound: 1781.7390604
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7390604, upper bound: 1781.7391794
time: 0.59 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -406.3652954, 1643.4683838, -436.2181396, 1764.3170166, -2170.6823730, 2079.6865234
1: -653.8065796, 1819.3048096, -701.2442017, 1953.7495117, -2607.5561523, 2520.5490723
2: -488.3619995, 2094.4963379, -523.9488525, 2249.0834961, -2737.4455566, 2618.4453125
3: -1050.2325439, 1875.6983643, -1127.9416504, 2012.0469971, -3062.2795410, 3003.6396484
4: -841.9771118, 1960.8830566, -903.7774048, 2103.8972168, -2945.8742676, 2864.6604004

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7391794, upper bound: 1781.7393295
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7391794, upper bound: 1781.7394478
time: 0.58 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.58 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 0, lower bound: -1781.7383546, upper bound: 1781.7385103
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 0, lower bound: -1781.7393295, upper bound: 1781.7391794
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 0, lower bound: -1781.7384967, upper bound: 1781.7387713
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 0, lower bound: -1781.7394478, upper bound: 1781.7394478
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 0, lower bound: -1781.7390604, upper bound: 1781.7390604
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 0, lower bound: -1781.7390604, upper bound: 1781.7391794
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 0, lower bound: -1781.7391794, upper bound: 1781.7393295
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 0, lower bound: -1781.7391794, upper bound: 1781.7394478

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -249.5951385, 1012.4692993, -364.9763794, 1475.6706543, -1725.2656250, 1377.4455566
1: -402.1629639, 1120.4627686, -587.3834839, 1633.7728271, -2035.9356689, 1707.8461914
2: -299.6103821, 1291.1125488, -438.7177429, 1881.2590332, -2180.8691406, 1729.8300781
3: -646.2778320, 1152.7542725, -943.6187134, 1684.4344482, -2330.7124023, 2096.3725586
4: -516.3628540, 1208.0513916, -756.9143677, 1760.8314209, -2277.1943359, 1964.9658203

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7364392, upper bound: 1781.7357519
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7365127, upper bound: 1781.7355336
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -280.9934998, 1140.6447754, -372.6616821, 1506.7690430, -1787.7624512, 1513.3062744
1: -452.9158020, 1262.5892334, -599.8692627, 1668.2287598, -2121.1442871, 1862.4584961
2: -337.9978638, 1453.7122803, -448.1130371, 1920.7075195, -2258.7053223, 1901.8253174
3: -726.7822266, 1300.5001221, -963.3588867, 1720.0947266, -2446.8767090, 2263.8588867
4: -582.2936401, 1360.9111328, -772.8338013, 1798.0191650, -2380.3122559, 2133.7448730

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7365810, upper bound: 1781.7365149
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7366495, upper bound: 1781.7361885
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -253.8181763, 1029.2421875, -418.5985413, 1692.8138428, -1946.6320801, 1447.8406982
1: -409.0047607, 1139.1433105, -672.6134644, 1874.6579590, -2283.6625977, 1811.7568359
2: -304.7549744, 1312.5236816, -502.5908203, 2157.9687500, -2462.7236328, 1815.1145020
3: -657.3359375, 1171.8070068, -1082.0802002, 1930.7851562, -2588.1208496, 2253.8864746
4: -525.1954346, 1227.9719238, -867.2494507, 2018.8156738, -2544.0104980, 2095.2207031

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7384967, upper bound: 1781.7387713
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7384967, upper bound: 1781.7387713
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -285.3908081, 1158.0187988, -426.4873047, 1724.4814453, -2009.8719482, 1584.5061035
1: -460.0427551, 1281.9267578, -685.4544678, 1909.7834473, -2369.8261719, 1967.3811035
2: -343.3165588, 1475.9349365, -512.2166138, 2198.1845703, -2541.5012207, 1988.1514893
3: -738.2792358, 1320.1748047, -1102.3721924, 1967.0922852, -2705.3715820, 2422.5463867
4: -591.4059448, 1381.5722656, -883.6120605, 2056.6840820, -2648.0895996, 2265.1843262

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7394478, upper bound: 1781.7394478
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7394478, upper bound: 1781.7394478
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -380.0692139, 1536.6132812, -381.5436096, 1542.9018555, -1922.9710693, 1918.1568604
1: -611.8825684, 1701.3682861, -614.2192383, 1708.1495361, -2320.0319824, 2315.5871582
2: -457.1301575, 1958.7752686, -458.8776855, 1966.8120117, -2423.9421387, 2417.6528320
3: -982.6779175, 1754.2751465, -986.3864746, 1761.2528076, -2743.9306641, 2740.6616211
4: -788.3563232, 1833.5460205, -791.3984375, 1840.9916992, -2629.3476562, 2624.9443359

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7380653, upper bound: 1781.7353029
time: 0.51 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7390604, upper bound: 1781.7390604
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -433.4168091, 1752.8944092, -381.5436096, 1542.9018555, -1976.3186035, 2134.4379883
1: -696.7171021, 1941.2100830, -614.2192383, 1708.1495361, -2404.8664551, 2555.4291992
2: -520.5569458, 2234.4389648, -458.8776855, 1966.8120117, -2487.3688965, 2693.3164062
3: -1120.6684570, 1999.0828857, -986.3864746, 1761.2528076, -2881.9213867, 2985.4692383
4: -897.9285889, 2090.1264648, -791.3984375, 1840.9916992, -2738.9199219, 2881.5249023

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7380653, upper bound: 1781.7354270
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7390604, upper bound: 1781.7391794
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -380.0692139, 1536.6132812, -436.2181396, 1764.3170166, -2144.3862305, 1972.8314209
1: -611.8825684, 1701.3682861, -701.2442017, 1953.7495117, -2565.6320801, 2402.6120605
2: -457.1301575, 1958.7752686, -523.9488525, 2249.0834961, -2706.2136230, 2482.7241211
3: -982.6779175, 1754.2751465, -1127.9416504, 2012.0469971, -2994.7248535, 2882.2160645
4: -788.3563232, 1833.5460205, -903.7774048, 2103.8972168, -2892.2529297, 2737.3234863

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7359351, upper bound: 1781.7366398
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7360567, upper bound: 1781.7365125
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -433.4168091, 1752.8944092, -436.2181396, 1764.3170166, -2197.7338867, 2189.1125488
1: -696.7171021, 1941.2100830, -701.2442017, 1953.7495117, -2650.4665527, 2642.4543457
2: -520.5569458, 2234.4389648, -523.9488525, 2249.0834961, -2769.6403809, 2758.3876953
3: -1120.6684570, 1999.0828857, -1127.9416504, 2012.0469971, -3132.7153320, 3127.0244141
4: -897.9285889, 2090.1264648, -903.7774048, 2103.8972168, -3001.8254395, 2993.9038086

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7359351, upper bound: 1781.7367697
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7360567, upper bound: 1781.7366447
time: 0.59 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.99 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -1781.7364392, upper bound: 1781.7357519
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -1781.7365127, upper bound: 1781.7355336
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -1781.7365810, upper bound: 1781.7365149
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -1781.7366495, upper bound: 1781.7361885
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -1781.7384967, upper bound: 1781.7387713
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -1781.7384967, upper bound: 1781.7387713
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -1781.7394478, upper bound: 1781.7394478
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -1781.7394478, upper bound: 1781.7394478
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -1781.7380653, upper bound: 1781.7353029
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -1781.7390604, upper bound: 1781.7390604
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -1781.7380653, upper bound: 1781.7354270
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -1781.7390604, upper bound: 1781.7391794
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -1781.7359351, upper bound: 1781.7366398
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -1781.7360567, upper bound: 1781.7365125
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -1781.7359351, upper bound: 1781.7367697
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -1781.7360567, upper bound: 1781.7366447

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -245.7764893, 997.1300659, -350.0047913, 1414.9172363, -1660.6937256, 1347.1348877
1: -396.0679932, 1103.4125977, -563.3749390, 1566.8156738, -1962.8835449, 1666.7874756
2: -295.0632019, 1271.5316162, -420.7089233, 1803.8642578, -2098.9274902, 1692.2404785
3: -636.6510010, 1134.8984375, -905.8338623, 1614.2926025, -2250.9436035, 2040.7321777
4: -508.3702087, 1189.7513428, -725.4912109, 1688.1950684, -2196.5649414, 1915.2423096

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7354329, upper bound: 1781.7343852
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7337758, upper bound: 1781.7325220
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7343822, upper bound: 1781.7335269
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -244.9183960, 993.6872559, -362.1889648, 1462.3582764, -1707.2766113, 1355.8762207
1: -394.6051025, 1099.6330566, -583.4194946, 1619.8314209, -2014.4364014, 1683.0524902
2: -293.9458008, 1267.1729736, -435.5825806, 1864.2728271, -2158.2187500, 1702.7556152
3: -634.2679443, 1130.9715576, -937.5336304, 1670.6684570, -2304.9362793, 2068.5051270
4: -506.6224060, 1185.5675049, -751.0195923, 1746.3682861, -2252.9907227, 1936.5870361

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7354354, upper bound: 1781.7330768
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7339789, upper bound: 1781.7324000
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7345612, upper bound: 1781.7334664
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -276.0041199, 1120.5465088, -357.5548096, 1445.4300537, -1721.4339600, 1478.1011963
1: -444.9812317, 1240.2762451, -575.7282104, 1600.5986328, -2045.5798340, 1816.0043945
2: -332.0309143, 1428.0793457, -429.9432068, 1842.5012207, -2174.5319824, 1858.0225830
3: -714.1357422, 1277.2164307, -925.2229004, 1649.2312012, -2363.3669434, 2202.4392090
4: -571.7902222, 1336.9530029, -741.0708008, 1724.6430664, -2296.4328613, 2078.0234375

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7355866, upper bound: 1781.7351689
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7363330, upper bound: 1781.7360096
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7363330, upper bound: 1781.7361886
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -277.3109131, 1125.6694336, -370.4865417, 1496.1790771, -1773.4899902, 1496.1558838
1: -446.9237976, 1246.0313721, -597.0407715, 1656.9739990, -2103.8977051, 1843.0721436
2: -333.5481262, 1434.6688232, -445.6776123, 1907.2895508, -2240.8374023, 1880.3464355
3: -717.3739014, 1283.1790771, -958.8373413, 1708.8659668, -2426.2392578, 2242.0163574
4: -574.6809082, 1343.0378418, -767.9973145, 1786.5903320, -2361.2712402, 2111.0351562

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7355866, upper bound: 1781.7339669
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7364431, upper bound: 1781.7360096
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7364431, upper bound: 1781.7361885
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -253.8181763, 1029.2421875, -353.4071045, 1427.2733154, -1681.0913086, 1382.6491699
1: -409.0047607, 1139.1433105, -569.3869019, 1581.7495117, -1990.7542725, 1708.5301514
2: -304.7549744, 1312.5236816, -425.3367310, 1820.1066895, -2124.8613281, 1737.8603516
3: -657.3359375, 1171.8070068, -916.9560547, 1628.3242188, -2285.6601562, 2088.7626953
4: -525.1954346, 1227.9719238, -732.9048462, 1704.1005859, -2229.2954102, 1960.8767090

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7356937, upper bound: 1781.7357108
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7365636, upper bound: 1781.7366486
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -253.8181763, 1029.2421875, -424.1564026, 1715.6539307, -1969.4721680, 1453.3981934
1: -409.0047607, 1139.1433105, -681.6588135, 1900.0026855, -2309.0073242, 1820.8021240
2: -304.7549744, 1312.5236816, -509.3135071, 2187.0554199, -2491.8100586, 1821.8371582
3: -657.3359375, 1171.8070068, -1096.8029785, 1956.6367188, -2613.9726562, 2268.6096191
4: -525.1954346, 1227.9719238, -878.8158569, 2045.6915283, -2570.8867188, 2106.7871094

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7356937, upper bound: 1781.7357108
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7365636, upper bound: 1781.7366486
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -285.3908081, 1158.0187988, -361.0393372, 1457.7868652, -1743.1776123, 1519.0581055
1: -460.0427551, 1281.9267578, -581.8721924, 1615.7539062, -2075.7963867, 1863.7989502
2: -343.3165588, 1475.9349365, -434.7326050, 1858.9564209, -2202.2727051, 1910.6674805
3: -738.2792358, 1320.1748047, -936.7194824, 1663.5185547, -2401.7976074, 2256.8942871
4: -591.4059448, 1381.5722656, -748.8656616, 1740.7049561, -2332.1108398, 2130.4379883

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7367697, upper bound: 1781.7364760
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7368000, upper bound: 1781.7366447
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -285.3908081, 1158.0187988, -432.0302429, 1747.2985840, -2032.6892090, 1590.0490723
1: -460.0427551, 1281.9267578, -694.5081787, 1935.1072998, -2395.1499023, 1976.4349365
2: -343.3165588, 1475.9349365, -518.9161987, 2227.2331543, -2570.5498047, 1994.8510742
3: -738.2792358, 1320.1748047, -1117.0438232, 1992.8728027, -2731.1518555, 2437.2185059
4: -591.4059448, 1381.5722656, -895.1329346, 2083.4797363, -2674.8852539, 2276.7050781

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7367697, upper bound: 1781.7364760
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7368000, upper bound: 1781.7366447
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -370.8344421, 1499.3065186, -341.6676025, 1380.7980957, -1751.6325684, 1840.9741211
1: -596.8895874, 1660.1257324, -550.2492676, 1528.7958984, -2125.6855469, 2210.3750000
2: -445.9160156, 1911.3458252, -410.5244446, 1760.7882080, -2206.7041016, 2321.8703613
3: -958.8629761, 1711.7945557, -884.1614990, 1575.3902588, -2534.2531738, 2595.9555664
4: -769.3325806, 1789.0432129, -708.3298950, 1647.8250732, -2417.1577148, 2497.3730469

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7354927, upper bound: 1781.7343245
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7352713, upper bound: 1781.7344561
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -378.5648499, 1530.5219727, -375.4452820, 1518.4366455, -1897.0013428, 1905.9672852
1: -609.4801636, 1694.7149658, -604.4939575, 1681.3911133, -2290.8706055, 2299.2089844
2: -455.3498230, 1950.9321289, -451.6637878, 1935.2952881, -2390.6450195, 2402.5959473
3: -978.7371216, 1747.5842285, -970.4394531, 1734.2108154, -2712.9477539, 2718.0236816
4: -785.3190308, 1826.3776855, -779.0832520, 1812.0585938, -2597.3776855, 2605.4609375

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7363846, upper bound: 1781.7359351
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7360567, upper bound: 1781.7360567
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -424.1564026, 1715.6539307, -341.6676025, 1380.7980957, -1804.9544678, 2057.3210449
1: -681.6588135, 1900.0026855, -550.2492676, 1528.7958984, -2210.4545898, 2450.2519531
2: -509.3135071, 2187.0554199, -410.5244446, 1760.7882080, -2270.1010742, 2597.5798340
3: -1096.8029785, 1956.6367188, -884.1614990, 1575.3902588, -2672.1931152, 2840.7983398
4: -878.8158569, 2045.6915283, -708.3298950, 1647.8250732, -2526.6408691, 2754.0214844

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7357340, upper bound: 1781.7344232
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7356935, upper bound: 1781.7345951
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -432.0302429, 1747.2985840, -375.4452820, 1518.4366455, -1950.4667969, 2122.7438965
1: -694.5081787, 1935.1072998, -604.4939575, 1681.3911133, -2375.8991699, 2539.6013184
2: -518.9161987, 2227.2331543, -451.6637878, 1935.2952881, -2454.2114258, 2678.8969727
3: -1117.0438232, 1992.8728027, -970.4394531, 1734.2108154, -2851.2541504, 2963.3120117
4: -895.1329346, 2083.4797363, -779.0832520, 1812.0585938, -2707.1914062, 2862.5625000

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7366398, upper bound: 1781.7360097
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7365125, upper bound: 1781.7361885
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -374.7412720, 1515.0307617, -419.6620789, 1696.8857422, -2071.6267090, 1934.6928711
1: -603.3477173, 1677.5599365, -674.7389526, 1879.4361572, -2482.7839355, 2352.2988281
2: -450.7215881, 1931.2829590, -504.0802002, 2163.1469727, -2613.8681641, 2435.3632812
3: -969.2091064, 1729.2766113, -1085.8531494, 1934.2558594, -2903.4648438, 2815.1298828
4: -777.1730347, 1807.7269287, -869.1304932, 2023.2640381, -2800.4370117, 2676.8571777

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7344232, upper bound: 1781.7357340
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7360097, upper bound: 1781.7366398
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -376.9759216, 1523.9462891, -434.9278870, 1756.6103516, -2133.5861816, 1958.8741455
1: -606.8365479, 1687.3646240, -699.6213989, 1945.8615723, -2552.6982422, 2386.9855957
2: -453.3874817, 1942.6628418, -522.4768066, 2239.5219727, -2692.9091797, 2465.1396484
3: -974.7493286, 1739.7324219, -1125.6015625, 2004.1170654, -2978.8664551, 2865.3330078
4: -781.9785156, 1818.4577637, -900.6910400, 2095.9714355, -2877.9497070, 2719.1489258

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7345951, upper bound: 1781.7356935
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7361885, upper bound: 1781.7365125
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -427.6822815, 1729.5706787, -419.6620789, 1696.8857422, -2124.5678711, 2149.2326660
1: -687.5330200, 1915.4976807, -674.7389526, 1879.4361572, -2566.9692383, 2590.2365723
2: -513.6772461, 2204.7358398, -504.0802002, 2163.1469727, -2676.8237305, 2708.8159180
3: -1106.0913086, 1972.1235352, -1085.8531494, 1934.2558594, -3040.3471680, 3057.9765625
4: -885.9403687, 2062.2402344, -869.1304932, 2023.2640381, -2909.2043457, 2931.3703613

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7364131, upper bound: 1781.7364131
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7364131, upper bound: 1781.7365745
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -430.1604614, 1739.7197266, -434.9278870, 1756.6103516, -2186.7705078, 2174.6477051
1: -691.4009399, 1926.5895996, -699.6213989, 1945.8615723, -2637.2624512, 2626.2102051
2: -516.6193848, 2217.6523438, -522.4768066, 2239.5219727, -2756.1411133, 2740.1291504
3: -1112.2281494, 1983.8931885, -1125.6015625, 2004.1170654, -3116.3452148, 3109.4941406
4: -891.2091675, 2074.3691406, -900.6910400, 2095.9714355, -2987.1804199, 2975.0600586

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7365739, upper bound: 1781.7364745
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7365739, upper bound: 1781.7366447
time: 0.61 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.64 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -1781.7337758, upper bound: 1781.7325220
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -1781.7343822, upper bound: 1781.7335269
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -1781.7339789, upper bound: 1781.7324000
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -1781.7345612, upper bound: 1781.7334664
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -1781.7363330, upper bound: 1781.7360096
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -1781.7363330, upper bound: 1781.7361886
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -1781.7364431, upper bound: 1781.7360096
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -1781.7364431, upper bound: 1781.7361885
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -1781.7356937, upper bound: 1781.7357108
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -1781.7365636, upper bound: 1781.7366486
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -1781.7356937, upper bound: 1781.7357108
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -1781.7365636, upper bound: 1781.7366486
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -1781.7367697, upper bound: 1781.7364760
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -1781.7368000, upper bound: 1781.7366447
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -1781.7367697, upper bound: 1781.7364760
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -1781.7368000, upper bound: 1781.7366447
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -1781.7354927, upper bound: 1781.7343245
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -1781.7352713, upper bound: 1781.7344561
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -1781.7363846, upper bound: 1781.7359351
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -1781.7360567, upper bound: 1781.7360567
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -1781.7357340, upper bound: 1781.7344232
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -1781.7356935, upper bound: 1781.7345951
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -1781.7366398, upper bound: 1781.7360097
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -1781.7365125, upper bound: 1781.7361885
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -1781.7344232, upper bound: 1781.7357340
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -1781.7360097, upper bound: 1781.7366398
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -1781.7345951, upper bound: 1781.7356935
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -1781.7361885, upper bound: 1781.7365125
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -1781.7364131, upper bound: 1781.7364131
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -1781.7364131, upper bound: 1781.7365745
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -1781.7365739, upper bound: 1781.7364745
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -1781.7365739, upper bound: 1781.7366447

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -232.9467773, 944.6403198, -347.5041504, 1404.6306152, -1637.5773926, 1292.1445312
1: -375.4278259, 1045.5295410, -559.3325806, 1555.4902344, -1930.9180908, 1604.8616943
2: -279.5939331, 1204.7821045, -417.6724854, 1790.7478027, -2070.3410645, 1622.4544678
3: -603.8522339, 1074.1973877, -899.3955078, 1602.5224609, -2206.3747559, 1973.5927734
4: -481.5502319, 1126.3951416, -720.2304077, 1675.8826904, -2157.4328613, 1846.6253662

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7320999, upper bound: 1781.7302878
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7330754, upper bound: 1781.7305997
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -243.3988342, 987.7400513, -348.7807617, 1409.9925537, -1653.3913574, 1336.5206299
1: -392.2226562, 1093.0083008, -561.3900757, 1561.3713379, -1953.5939941, 1654.3983154
2: -292.1804504, 1259.5788574, -419.2455444, 1797.6016846, -2089.7822266, 1678.8243408
3: -630.4731445, 1124.0252686, -902.6779785, 1608.6208496, -2239.0939941, 2026.7032471
4: -503.4412231, 1178.4139404, -722.9876099, 1682.2587891, -2185.6999512, 1901.4013672

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7327784, upper bound: 1781.7312625
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7337589, upper bound: 1781.7316361
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -232.2640533, 941.7647095, -359.0104065, 1449.3765869, -1681.6402588, 1300.7750244
1: -374.2321167, 1042.3763428, -578.3112183, 1605.5147705, -1979.7468262, 1620.6875000
2: -278.6949463, 1201.1577148, -431.7445679, 1847.7498779, -2126.4448242, 1632.9023438
3: -601.9019165, 1070.9119873, -929.3821411, 1655.7640381, -2257.6655273, 2000.2939453
4: -480.1738586, 1122.9510498, -744.3676758, 1730.8089600, -2210.9826660, 1867.3186035

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7327749, upper bound: 1781.7311926
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7327749, upper bound: 1781.7324000
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -242.4387970, 983.7369385, -360.6931152, 1456.1618652, -1698.6004639, 1344.4300537
1: -390.5806274, 1088.6379395, -580.9852905, 1613.0615234, -2003.6420898, 1669.6232910
2: -290.9404602, 1254.5075684, -433.7901917, 1856.4025879, -2147.3430176, 1688.2977295
3: -627.8339844, 1119.4835205, -933.6999512, 1663.5633545, -2291.3974609, 2053.1835938
4: -501.4797974, 1173.6070557, -747.9345093, 1738.9378662, -2240.4172363, 1921.5415039

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7337848, upper bound: 1781.7321369
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7337848, upper bound: 1781.7334664
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -266.2103271, 1080.7606201, -357.5548096, 1445.4300537, -1711.6402588, 1438.3154297
1: -429.4125061, 1196.1322021, -575.7282104, 1600.5986328, -2030.0111084, 1771.8603516
2: -320.3195496, 1377.4125977, -429.9432068, 1842.5012207, -2162.8205566, 1807.3558350
3: -689.2888794, 1231.2491455, -925.2229004, 1649.2312012, -2338.5200195, 2156.4721680
4: -551.1499023, 1289.6032715, -741.0708008, 1724.6430664, -2275.7927246, 2030.6740723

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7339063, upper bound: 1781.7346080
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7346046, upper bound: 1781.7344362
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -284.9710999, 1152.5366211, -357.5548096, 1445.4300537, -1730.4008789, 1510.0914307
1: -459.8998718, 1275.6610107, -575.7282104, 1600.5986328, -2060.4985352, 1851.3891602
2: -342.9600830, 1469.4373779, -429.9432068, 1842.5012207, -2185.4609375, 1899.3806152
3: -737.6937256, 1314.7252197, -925.2229004, 1649.2312012, -2386.9248047, 2239.9482422
4: -589.6637573, 1376.7943115, -741.0708008, 1724.6430664, -2314.3063965, 2117.8652344

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7339063, upper bound: 1781.7347346
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7346046, upper bound: 1781.7346363
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -266.2103271, 1080.7606201, -370.4865417, 1496.1790771, -1762.3894043, 1451.2471924
1: -429.4125061, 1196.1322021, -597.0407715, 1656.9739990, -2086.3864746, 1793.1729736
2: -320.3195496, 1377.4125977, -445.6776123, 1907.2895508, -2227.6091309, 1823.0902100
3: -689.2888794, 1231.2491455, -958.8373413, 1708.8659668, -2398.1545410, 2190.0861816
4: -551.1499023, 1289.6032715, -767.9973145, 1786.5903320, -2337.7402344, 2057.6000977

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7358744, upper bound: 1781.7353317
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7362280, upper bound: 1781.7358620
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -284.9710999, 1152.5366211, -370.4865417, 1496.1790771, -1781.1500244, 1523.0230713
1: -459.8998718, 1275.6610107, -597.0407715, 1656.9739990, -2116.8737793, 1872.7017822
2: -342.9600830, 1469.4373779, -445.6776123, 1907.2895508, -2250.2495117, 1915.1149902
3: -737.6937256, 1314.7252197, -958.8373413, 1708.8659668, -2446.5595703, 2273.5620117
4: -589.6637573, 1376.7943115, -767.9973145, 1786.5903320, -2376.2539062, 2144.7915039

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7358744, upper bound: 1781.7355861
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7362280, upper bound: 1781.7360031
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -240.5637512, 975.2540283, -349.4981384, 1411.4128418, -1651.9764404, 1324.7521973
1: -387.6956787, 1079.4962158, -563.0755005, 1564.2469482, -1951.9426270, 1642.5717773
2: -288.8030396, 1243.8601074, -420.6538086, 1799.9036865, -2088.7067871, 1664.5139160
3: -623.3930664, 1109.2934570, -906.8986206, 1610.1361084, -2233.5292969, 2016.1920166
4: -497.4926758, 1162.9390869, -724.7752686, 1685.0428467, -2182.5354004, 1887.7142334

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7338664, upper bound: 1781.7327552
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7341295, upper bound: 1781.7331953
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -250.4854889, 1016.0076294, -352.6181641, 1424.0621338, -1674.5474854, 1368.6257324
1: -403.5925903, 1124.5312500, -568.1134644, 1578.2065430, -1981.7989502, 1692.6447754
2: -300.7073975, 1295.6479492, -424.3776550, 1816.0239258, -2116.7314453, 1720.0252686
3: -648.6723633, 1156.6143799, -914.9185181, 1624.6082764, -2273.2797852, 2071.5329590
4: -518.3111572, 1211.9686279, -731.2630005, 1700.2308350, -2218.5419922, 1943.2316895

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7344814, upper bound: 1781.7337564
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7347222, upper bound: 1781.7341393
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -240.5637512, 975.2540283, -421.1238098, 1703.2177734, -1943.7814941, 1396.3778076
1: -387.6956787, 1079.4962158, -676.7351685, 1886.3051758, -2274.0009766, 1756.2312012
2: -288.8030396, 1243.8601074, -505.6578369, 2171.2053223, -2460.0083008, 1749.5179443
3: -623.3930664, 1109.2934570, -1088.9458008, 1942.3984375, -2565.7915039, 2198.2392578
4: -497.4926758, 1162.9390869, -872.4973145, 2030.7888184, -2528.2805176, 2035.4362793

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7338664, upper bound: 1781.7327552
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7341295, upper bound: 1781.7328856
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -250.4854889, 1016.0076294, -422.5335999, 1708.9998779, -1959.4852295, 1438.5412598
1: -403.5925903, 1124.5312500, -679.0330811, 1892.7042236, -2296.2968750, 1803.5642090
2: -300.7073975, 1295.6479492, -507.3629456, 2178.6040039, -2479.3115234, 1803.0108643
3: -648.6723633, 1156.6143799, -1092.6533203, 1948.9844971, -2597.6567383, 2249.2675781
4: -518.3111572, 1211.9686279, -875.4647217, 2037.7102051, -2556.0214844, 2087.4331055

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7344814, upper bound: 1781.7337564
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7347216, upper bound: 1781.7339260
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -270.0743408, 1096.1451416, -355.9144897, 1437.1628418, -1707.2369385, 1452.0594482
1: -435.7109680, 1213.2921143, -573.6865234, 1592.9003906, -2028.6112061, 1786.9786377
2: -325.0308533, 1397.0957031, -428.5954590, 1832.6630859, -2157.6938477, 1825.6911621
3: -699.4786987, 1248.7093506, -923.6848145, 1639.6293945, -2339.1081543, 2172.3940430
4: -559.2321167, 1307.8543701, -738.1409302, 1716.0385742, -2275.2700195, 2045.9948730

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7364131, upper bound: 1781.7364131
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7364131, upper bound: 1781.7366652
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -287.6105042, 1163.5479736, -357.0361938, 1441.6520996, -1729.2624512, 1520.5838623
1: -464.1067200, 1287.8089600, -575.3914795, 1597.8937988, -2062.0002441, 1863.2004395
2: -346.1171265, 1483.4461670, -429.8942566, 1838.3767090, -2184.4938965, 1913.3402100
3: -744.4826050, 1327.0776367, -926.4432373, 1644.8979492, -2389.3798828, 2253.5207520
4: -595.1425171, 1389.7296143, -740.5503540, 1721.3214111, -2316.4638672, 2130.2800293

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7366652, upper bound: 1781.7365746
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7366652, upper bound: 1781.7368000
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -270.0743408, 1096.1451416, -426.2690735, 1723.8718262, -1993.9460449, 1522.4141846
1: -435.7109680, 1213.2921143, -685.2836304, 1909.2832031, -2344.9941406, 1898.5756836
2: -325.0308533, 1397.0957031, -512.0061035, 2197.3974609, -2522.4282227, 1909.1018066
3: -699.4786987, 1248.7093506, -1102.4012451, 1965.7993164, -2665.2775879, 2351.1105957
4: -559.2321167, 1307.8543701, -883.0895386, 2055.4760742, -2614.7077637, 2190.9438477

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7364131, upper bound: 1781.7364131
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7364131, upper bound: 1781.7364760
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -287.6105042, 1163.5479736, -428.7889709, 1734.1849365, -2021.7954102, 1592.3367920
1: -464.1067200, 1287.8089600, -689.2094727, 1920.5543213, -2384.6604004, 1977.0184326
2: -346.1171265, 1483.4461670, -514.9960327, 2210.5244141, -2556.6416016, 1998.4418945
3: -744.4826050, 1327.0776367, -1108.6405029, 1977.7557373, -2722.2377930, 2435.7180176
4: -595.1425171, 1389.7296143, -888.4437256, 2067.7976074, -2662.9399414, 2278.1728516

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7366652, upper bound: 1781.7365746
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7366652, upper bound: 1781.7366447
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -355.9729309, 1439.0855713, -337.3941040, 1363.3648682, -1719.3377686, 1776.4797363
1: -573.0660400, 1593.6895752, -543.4124146, 1509.5671387, -2082.6330566, 2137.1010742
2: -428.0492859, 1834.6171875, -405.3940735, 1738.6140137, -2166.6633301, 2240.0109863
3: -921.3684692, 1642.0953369, -873.4146729, 1555.1976318, -2476.5661621, 2515.5095215
4: -738.1330566, 1717.0083008, -699.3547363, 1627.0383301, -2365.1706543, 2416.3627930

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7341120, upper bound: 1781.7329561
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7331460, upper bound: 1781.7297069
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -368.6636047, 1488.9306641, -337.7725830, 1364.9427490, -1733.6063232, 1826.7031250
1: -593.8600464, 1649.0731201, -543.9396973, 1511.2423096, -2105.1022949, 2193.0126953
2: -443.4406738, 1898.1647949, -405.7970276, 1740.5892334, -2184.0297852, 2303.9619141
3: -954.1411743, 1701.0026855, -874.1314087, 1557.1479492, -2511.2890625, 2575.1340332
4: -764.5875244, 1778.0478516, -700.2363281, 1628.9226074, -2393.5097656, 2478.2832031

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7328458, upper bound: 1781.7330288
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7318812, upper bound: 1781.7297733
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -363.6766052, 1470.1733398, -370.1067505, 1496.8090820, -1860.4857178, 1840.2800293
1: -585.6828613, 1628.1141357, -595.9548340, 1657.5401611, -2243.2229004, 2224.0688477
2: -437.4640198, 1873.9901123, -445.2465820, 1907.7340088, -2345.1977539, 2319.2368164
3: -941.1439819, 1677.7342529, -956.9501343, 1709.1711426, -2650.3144531, 2634.6843262
4: -754.0519409, 1754.1743164, -767.8624268, 1786.2060547, -2540.2580566, 2522.0366211

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7350380, upper bound: 1781.7351114
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7342559, upper bound: 1781.7330038
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -377.1117859, 1523.2244873, -372.3499756, 1505.7174072, -1882.8291016, 1895.5743408
1: -607.6929321, 1686.7650146, -599.4376831, 1667.3337402, -2275.0266113, 2286.2026367
2: -453.7145081, 1941.7670898, -447.9152832, 1919.1092529, -2372.8237305, 2389.6823730
3: -975.7833252, 1739.8317871, -962.4932861, 1719.6199951, -2695.4030762, 2702.3251953
4: -781.8900757, 1818.8814697, -772.6890869, 1796.9195557, -2578.8085938, 2591.5703125

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7338196, upper bound: 1781.7351202
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7330869, upper bound: 1781.7330869
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -407.6321411, 1648.3116455, -337.3941040, 1363.3648682, -1770.9968262, 1985.7056885
1: -655.1989746, 1825.8134766, -543.4124146, 1509.5671387, -2164.7661133, 2369.2253418
2: -489.4905396, 2101.2966309, -405.3940735, 1738.6140137, -2228.1044922, 2506.6906738
3: -1054.8006592, 1878.9331055, -873.4146729, 1555.1976318, -2609.9982910, 2752.3471680
4: -844.2623291, 1965.2430420, -699.3547363, 1627.0383301, -2471.2998047, 2664.5976562

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7351925, upper bound: 1781.7340527
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7350469, upper bound: 1781.7339187
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7331902, upper bound: 1781.7327474
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7331110, upper bound: 1781.7325245
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -422.7800598, 1707.2832031, -337.7725830, 1364.9427490, -1787.7226562, 2045.0556641
1: -679.8095093, 1891.5803223, -543.9396973, 1511.2423096, -2191.0517578, 2435.5200195
2: -507.7368164, 2176.6662598, -405.7970276, 1740.5892334, -2248.3259277, 2582.4633789
3: -1094.1484375, 1948.2749023, -874.1314087, 1557.1479492, -2651.2963867, 2822.4062500
4: -875.6743164, 2037.1665039, -700.2363281, 1628.9226074, -2504.5969238, 2737.4025879

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7351908, upper bound: 1781.7342000
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7344178, upper bound: 1781.7340041
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7329913, upper bound: 1781.7329111
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7329133, upper bound: 1781.7326881
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -415.4491882, 1679.8620605, -370.1067505, 1496.8090820, -1912.2583008, 2049.9685059
1: -667.9820557, 1860.7307129, -595.9548340, 1657.5401611, -2325.5217285, 2456.6855469
2: -499.0282898, 2141.3107910, -445.2465820, 1907.7340088, -2406.7622070, 2586.5573730
3: -1074.8863525, 1915.0198975, -956.9501343, 1709.1711426, -2784.0566406, 2871.9697266
4: -860.4401855, 2002.8547363, -767.8624268, 1786.2060547, -2646.6459961, 2770.7167969

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7363330, upper bound: 1781.7360097
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7363330, upper bound: 1781.7360097
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -431.2170410, 1741.7592773, -372.3499756, 1505.7174072, -1936.9344482, 2114.1093750
1: -693.6503296, 1929.4581299, -599.4376831, 1667.3337402, -2360.9841309, 2528.8955078
2: -517.9996338, 2220.4948730, -447.9152832, 1919.1092529, -2437.1088867, 2668.4101562
3: -1115.9855957, 1987.3259277, -962.4932861, 1719.6199951, -2835.6054688, 2949.8190918
4: -893.0153198, 2078.2082520, -772.6890869, 1796.9195557, -2689.9338379, 2850.8974609

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7363986, upper bound: 1781.7361885
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7363986, upper bound: 1781.7361885
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -336.1095276, 1357.7355957, -410.4321594, 1659.7413330, -1995.8507080, 1768.1677246
1: -541.3595581, 1503.5460205, -659.7197876, 1838.3608398, -2379.7204590, 2163.2658691
2: -403.8824158, 1731.4285889, -492.8752136, 2115.9401855, -2519.8222656, 2224.3037109
3: -870.2057495, 1548.9615479, -1062.0689697, 1891.9353027, -2762.1408691, 2611.0305176
4: -696.7113647, 1620.3636475, -850.0967407, 1979.0159912, -2675.7272949, 2470.4604492

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7344232, upper bound: 1781.7357340
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7344232, upper bound: 1781.7357340
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -368.5567017, 1490.1575928, -418.2988892, 1691.4465332, -2060.0031738, 1908.4565430
1: -593.4999390, 1650.3736572, -672.5762939, 1873.4897461, -2466.9895020, 2322.9494629
2: -443.4110718, 1899.2469482, -502.4725342, 2156.1274414, -2599.5385742, 2401.7189941
3: -953.0557251, 1701.7620850, -1082.2894287, 1928.2283936, -2881.2839355, 2784.0515137
4: -764.6622925, 1778.3319092, -866.3833618, 2016.8157959, -2781.4780273, 2644.7153320

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7360097, upper bound: 1781.7366398
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7360097, upper bound: 1781.7366398
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -336.5470886, 1359.5643311, -425.3000183, 1717.4219971, -2053.9689941, 1784.8643799
1: -541.9805908, 1505.4973145, -683.8845825, 1902.7894287, -2444.7700195, 2189.3818359
2: -404.3509521, 1733.7231445, -510.7950745, 2189.6367188, -2593.9875488, 2244.5180664
3: -871.0684204, 1551.2114258, -1100.7252197, 1959.8452148, -2830.9135742, 2651.9365234
4: -697.7113037, 1622.5474854, -880.9306030, 2049.3977051, -2747.1088867, 2503.4780273

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7345951, upper bound: 1781.7356935
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7345951, upper bound: 1781.7356935
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -370.8687134, 1499.3652344, -433.7557373, 1751.9511719, -2122.8198242, 1933.1207275
1: -597.0952148, 1660.4860840, -697.7598267, 1940.7407227, -2537.8359375, 2358.2456055
2: -446.1637573, 1911.0021973, -521.0839844, 2233.5366211, -2679.6997070, 2432.0859375
3: -958.7696533, 1712.5645752, -1122.5920410, 1998.9608154, -2957.7304688, 2835.1567383
4: -769.6380615, 1789.4100342, -898.3148193, 2090.4948730, -2860.1328125, 2687.7248535

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7361885, upper bound: 1781.7365125
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7361885, upper bound: 1781.7365125
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -416.8405151, 1685.3878174, -419.6620789, 1696.8857422, -2113.7260742, 2105.0495605
1: -670.1839600, 1866.7962646, -674.7389526, 1879.4361572, -2549.6201172, 2541.5351562
2: -500.6704102, 2148.4094238, -504.0802002, 2163.1469727, -2663.8168945, 2652.4897461
3: -1078.5261230, 1921.1632080, -1085.8531494, 1934.2558594, -3012.7819824, 3007.0156250
4: -863.2470093, 2009.3989258, -869.1304932, 2023.2640381, -2886.5109863, 2878.5292969

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7364131, upper bound: 1781.7365981
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7364131, upper bound: 1781.7365981
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -432.3953552, 1746.4383545, -419.6620789, 1696.8857422, -2129.2810059, 2166.1003418
1: -695.5231323, 1934.6020508, -674.7389526, 1879.4361572, -2574.9592285, 2609.3405762
2: -519.4005127, 2226.5065918, -504.0802002, 2163.1469727, -2682.5468750, 2730.5869141
3: -1119.0119629, 1992.5078125, -1085.8531494, 1934.2558594, -3053.2678223, 3078.3608398
4: -895.4044189, 2083.7121582, -869.1304932, 2023.2640381, -2918.6684570, 2952.8422852

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7364131, upper bound: 1781.7367697
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7364131, upper bound: 1781.7367697
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -416.8405151, 1685.3878174, -434.9278870, 1756.6103516, -2173.4504395, 2120.3151855
1: -670.1839600, 1866.7962646, -699.6213989, 1945.8615723, -2616.0451660, 2566.4174805
2: -500.6704102, 2148.4094238, -522.4768066, 2239.5219727, -2740.1918945, 2670.8862305
3: -1078.5261230, 1921.1632080, -1125.6015625, 2004.1170654, -3082.6430664, 3046.7634277
4: -863.2470093, 2009.3989258, -900.6910400, 2095.9714355, -2959.2177734, 2910.0898438

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7364131, upper bound: 1781.7364745
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7364131, upper bound: 1781.7364745
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -432.3953552, 1746.4383545, -434.9278870, 1756.6103516, -2189.0056152, 2181.3662109
1: -695.5231323, 1934.6020508, -699.6213989, 1945.8615723, -2641.3847656, 2634.2226562
2: -519.4005127, 2226.5065918, -522.4768066, 2239.5219727, -2758.9218750, 2748.9833984
3: -1119.0119629, 1992.5078125, -1125.6015625, 2004.1170654, -3123.1289062, 3118.1091309
4: -895.4044189, 2083.7121582, -900.6910400, 2095.9714355, -2991.3752441, 2984.4025879

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7364131, upper bound: 1781.7366447
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7364131, upper bound: 1781.7366447
time: 0.57 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.86 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7320999, upper bound: 1781.7302878
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7330754, upper bound: 1781.7305997
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7327784, upper bound: 1781.7312625
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7337589, upper bound: 1781.7316361
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7327749, upper bound: 1781.7311926
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7327749, upper bound: 1781.7324000
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7337848, upper bound: 1781.7321369
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7337848, upper bound: 1781.7334664
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7339063, upper bound: 1781.7346080
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7346046, upper bound: 1781.7344362
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7339063, upper bound: 1781.7347346
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7346046, upper bound: 1781.7346363
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7358744, upper bound: 1781.7353317
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7362280, upper bound: 1781.7358620
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7358744, upper bound: 1781.7355861
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7362280, upper bound: 1781.7360031
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7338664, upper bound: 1781.7327552
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7341295, upper bound: 1781.7331953
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7344814, upper bound: 1781.7337564
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7347222, upper bound: 1781.7341393
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7338664, upper bound: 1781.7327552
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7341295, upper bound: 1781.7328856
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7344814, upper bound: 1781.7337564
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7347216, upper bound: 1781.7339260
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7364131, upper bound: 1781.7364131
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7364131, upper bound: 1781.7366652
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7366652, upper bound: 1781.7365746
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7366652, upper bound: 1781.7368000
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7364131, upper bound: 1781.7364131
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7364131, upper bound: 1781.7364760
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7366652, upper bound: 1781.7365746
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7366652, upper bound: 1781.7366447
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7341120, upper bound: 1781.7329561
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7331460, upper bound: 1781.7297069
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7328458, upper bound: 1781.7330288
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7318812, upper bound: 1781.7297733
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7350380, upper bound: 1781.7351114
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7342559, upper bound: 1781.7330038
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7338196, upper bound: 1781.7351202
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7330869, upper bound: 1781.7330869
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7331902, upper bound: 1781.7327474
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7331110, upper bound: 1781.7325245
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7329913, upper bound: 1781.7329111
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7329133, upper bound: 1781.7326881
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7363330, upper bound: 1781.7360097
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7363330, upper bound: 1781.7360097
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7363986, upper bound: 1781.7361885
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7363986, upper bound: 1781.7361885
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7344232, upper bound: 1781.7357340
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7344232, upper bound: 1781.7357340
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7360097, upper bound: 1781.7366398
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7360097, upper bound: 1781.7366398
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7345951, upper bound: 1781.7356935
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7345951, upper bound: 1781.7356935
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7361885, upper bound: 1781.7365125
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7361885, upper bound: 1781.7365125
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7364131, upper bound: 1781.7365981
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7364131, upper bound: 1781.7365981
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7364131, upper bound: 1781.7367697
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7364131, upper bound: 1781.7367697
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7364131, upper bound: 1781.7364745
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7364131, upper bound: 1781.7364745
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7364131, upper bound: 1781.7366447
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.86
Output dim: 0, lower bound: -1781.7364131, upper bound: 1781.7366447

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -224.0402985, 908.9295654, -321.5458069, 1304.5616455, -1528.6019287, 1230.4752197
1: -361.3619080, 1005.6787720, -518.5918579, 1441.7990723, -1803.1608887, 1524.2703857
2: -269.0679932, 1159.2094727, -386.7334595, 1663.9616699, -1933.0294189, 1545.9427490
3: -581.4100342, 1033.0152588, -832.4993286, 1484.2965088, -2065.7065430, 1865.5144043
4: -463.0280457, 1084.0075684, -666.2349243, 1555.3416748, -2018.3697510, 1750.2424316

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7303536, upper bound: 1781.7276324
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7314855, upper bound: 1781.7297189
time: 0.50 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7306139, upper bound: 1781.7275385
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 25

Time for candidate selection: 7.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7319193, upper bound: 1781.7288549
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7318042, upper bound: 1781.7302878
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7267042, upper bound: 1781.7286597
time: 0.50 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7272808, upper bound: 1781.7279996
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7271373, upper bound: 1781.7278567
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -231.2093506, 937.6601562, -342.4534912, 1384.0546875, -1615.2637939, 1280.1136475
1: -372.6014099, 1037.7830811, -551.1367798, 1532.7689209, -1905.3702393, 1588.9199219
2: -277.4645996, 1195.8763428, -411.4872131, 1764.5202637, -2041.9846191, 1607.3635254
3: -599.3376465, 1066.1511230, -886.3994141, 1578.9835205, -2178.3212891, 1952.5504150
4: -477.9020691, 1118.0338135, -709.6556396, 1651.3107910, -2129.2128906, 1827.6894531

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7309883, upper bound: 1781.7269713
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7326889, upper bound: 1781.7293654
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7317660, upper bound: 1781.7277821
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 25

Time for candidate selection: 7.27 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7330754, upper bound: 1781.7301031
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7324235, upper bound: 1781.7304183
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7278499, upper bound: 1781.7280601
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7226574, upper bound: 1781.7225885
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7238920, upper bound: 1781.7234982
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -234.3899994, 951.3995361, -322.7827148, 1309.5936279, -1543.9835205, 1274.1822510
1: -377.9932556, 1052.5595703, -520.5691528, 1447.3437500, -1825.3366699, 1573.1286621
2: -281.5419922, 1213.2633057, -388.2387695, 1670.3898926, -1951.9318848, 1601.5020752
3: -607.8390503, 1082.2142334, -835.6818848, 1490.0231934, -2097.8623047, 1917.8961182
4: -484.7222900, 1135.3029785, -668.8378296, 1561.3684082, -2046.0905762, 1804.1403809

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7318474, upper bound: 1781.7302711
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7326620, upper bound: 1781.7308468
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7325691, upper bound: 1781.7308750
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -241.5305939, 980.2428589, -343.7533875, 1389.4987793, -1631.0292969, 1323.9960938
1: -389.1887817, 1084.7015381, -553.2338257, 1538.7370605, -1927.9256592, 1637.9353027
2: -289.8886719, 1250.0228271, -413.0894775, 1771.4796143, -2061.3679199, 1663.1123047
3: -625.6351318, 1115.3787842, -889.7370605, 1585.1892090, -2210.8242188, 2005.1158447
4: -499.5135193, 1169.4208984, -712.4605713, 1657.8023682, -2157.3156738, 1881.8814697

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7328512, upper bound: 1781.7305335
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7336402, upper bound: 1781.7314139
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7335364, upper bound: 1781.7314304
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -232.2640533, 941.7647095, -326.3352051, 1317.2365723, -1549.5002441, 1268.0998535
1: -374.2321167, 1042.3763428, -525.6558228, 1458.7736816, -1833.0058594, 1568.0322266
2: -278.6949463, 1201.1577148, -392.0762634, 1679.5454102, -1958.2402344, 1593.2340088
3: -601.9019165, 1070.9119873, -845.2996826, 1504.0223389, -2105.9240723, 1916.2114258
4: -480.1738586, 1122.9510498, -676.0542603, 1573.5794678, -2053.7534180, 1799.0051270

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7279806, upper bound: 1781.7237762
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7315882, upper bound: 1781.7292401
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 25

Time for candidate selection: 6.36 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7327749, upper bound: 1781.7311926
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7214729, upper bound: 1781.7198587
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7227848, upper bound: 1781.7207581
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -232.2640533, 941.7647095, -363.6779175, 1468.1951904, -1700.4591064, 1305.4426270
1: -374.2321167, 1042.3763428, -585.9694824, 1626.1545410, -2000.3867188, 1628.3458252
2: -278.6949463, 1201.1577148, -437.4189453, 1871.4951172, -2150.1896973, 1638.5766602
3: -601.9019165, 1070.9119873, -940.8806763, 1677.4216309, -2279.3234863, 2011.7923584
4: -480.1738586, 1122.9510498, -753.7445679, 1753.4254150, -2233.5993652, 1876.6955566

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7279806, upper bound: 1781.7273002
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7315882, upper bound: 1781.7300496
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 25

Time for candidate selection: 6.54 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7327749, upper bound: 1781.7324000
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7214729, upper bound: 1781.7229029
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7227848, upper bound: 1781.7237885
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -242.4387970, 983.7369385, -327.9824524, 1323.8323975, -1566.2711182, 1311.7193604
1: -390.5806274, 1088.6379395, -528.2642212, 1466.1353760, -1856.7160645, 1616.9020996
2: -290.9404602, 1254.5075684, -394.0757446, 1687.9614258, -1978.9018555, 1648.5832520
3: -627.8339844, 1119.4835205, -849.5301514, 1511.6403809, -2139.4743652, 1969.0135498
4: -501.4797974, 1173.6070557, -679.5574341, 1581.4844971, -2082.9643555, 1853.1644287

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7321825, upper bound: 1781.7286187
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7321199, upper bound: 1781.7303678
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 25

Time for candidate selection: 6.63 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7337848, upper bound: 1781.7321369
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7337848, upper bound: 1781.7321369
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -242.4387970, 983.7369385, -365.4122314, 1475.1975098, -1717.6361084, 1349.1491699
1: -390.5806274, 1088.6379395, -588.7369995, 1633.9444580, -2024.5251465, 1677.3750000
2: -290.9404602, 1254.5075684, -439.5350647, 1880.4177246, -2171.3581543, 1694.0426025
3: -627.8339844, 1119.4835205, -945.3305664, 1685.4655762, -2313.2995605, 2064.8137207
4: -501.4797974, 1173.6070557, -757.4373169, 1761.7962646, -2263.2761230, 1931.0444336

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7321825, upper bound: 1781.7313810
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7321199, upper bound: 1781.7311736
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 25

Time for candidate selection: 6.59 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7337848, upper bound: 1781.7334664
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7337848, upper bound: 1781.7334664
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -256.8348999, 1042.4641113, -330.2367249, 1339.6118164, -1596.4464111, 1372.7008057
1: -414.5786133, 1153.4822998, -532.5385132, 1480.7984619, -1895.3770752, 1686.0206299
2: -309.1516418, 1328.7788086, -397.1693726, 1708.5131836, -2017.6647949, 1725.9482422
3: -665.4036255, 1187.2767334, -854.9274902, 1524.6968994, -2190.1005859, 2042.2042236
4: -531.4086914, 1244.2235107, -684.3079834, 1597.3702393, -2128.7785645, 1928.5314941

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7331238, upper bound: 1781.7333452
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7311290, upper bound: 1781.7312896
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7314278, upper bound: 1781.7319360
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -264.5609436, 1074.1499023, -352.5554199, 1425.1358643, -1689.6967773, 1426.7052002
1: -426.7315369, 1188.8284912, -567.5919189, 1578.1818848, -2004.9134521, 1756.4202881
2: -318.3057861, 1368.9805908, -423.8333740, 1816.6397705, -2134.9453125, 1792.8139648
3: -685.0457153, 1223.6311035, -912.3572388, 1626.0194092, -2311.0651855, 2135.9882812
4: -547.7048340, 1281.6756592, -730.6228027, 1700.4152832, -2248.1201172, 2012.2979736

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7339048, upper bound: 1781.7331904
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7320185, upper bound: 1781.7315161
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7323331, upper bound: 1781.7321740
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -274.7355652, 1110.5478516, -330.2367249, 1339.6118164, -1614.3474121, 1440.7845459
1: -443.4833984, 1228.7630615, -532.5385132, 1480.7984619, -1924.2817383, 1761.3015137
2: -330.6237488, 1416.0394287, -397.1693726, 1708.5131836, -2039.1369629, 1813.2087402
3: -711.3191528, 1266.3244629, -854.9274902, 1524.6968994, -2236.0161133, 2121.2517090
4: -567.9056396, 1326.9449463, -684.3079834, 1597.3702393, -2165.2758789, 2011.2529297

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7338351, upper bound: 1781.7334964
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7320633, upper bound: 1781.7314030
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7326668, upper bound: 1781.7321477
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -283.7110596, 1147.4442139, -352.5554199, 1425.1358643, -1708.8469238, 1499.9995117
1: -457.8697510, 1270.0161133, -567.5919189, 1578.1818848, -2036.0516357, 1837.6080322
2: -341.4356995, 1462.9710693, -423.8333740, 1816.6397705, -2158.0754395, 1886.8044434
3: -734.4865112, 1308.8919678, -912.3572388, 1626.0194092, -2360.5058594, 2221.2492676
4: -587.0469360, 1370.7595215, -730.6228027, 1700.4152832, -2287.4619141, 2101.3823242

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7346951, upper bound: 1781.7333868
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7330189, upper bound: 1781.7316629
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7335917, upper bound: 1781.7324391
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -265.4252319, 1077.6252441, -365.4256592, 1476.6849365, -1742.1101074, 1443.0509033
1: -428.1300964, 1192.6667480, -588.9390259, 1634.8612061, -2062.9912109, 1781.6057129
2: -319.3626099, 1373.4160156, -439.4110107, 1882.3299561, -2201.6921387, 1812.8270264
3: -687.2263184, 1227.6256104, -945.5795288, 1685.5665283, -2372.7927246, 2173.2050781
4: -549.5196533, 1285.7872314, -756.9507446, 1762.7757568, -2312.2949219, 2042.7379150

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7356100, upper bound: 1781.7335421
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7336903, upper bound: 1781.7326368
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7339514, upper bound: 1781.7333169
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -264.9101562, 1075.6405029, -366.4806824, 1479.8315430, -1744.7416992, 1442.1210938
1: -427.3601379, 1190.4272461, -590.6148682, 1638.8887939, -2066.2490234, 1781.0421143
2: -318.7637329, 1370.8758545, -440.8607483, 1886.4167480, -2205.1804199, 1811.7365723
3: -685.9980469, 1225.2420654, -948.4612427, 1689.8979492, -2375.8957520, 2173.7031250
4: -548.3972778, 1283.4224854, -759.5183716, 1766.9053955, -2315.3027344, 2042.9405518

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7353605, upper bound: 1781.7330563
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7339385, upper bound: 1781.7330318
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7342025, upper bound: 1781.7337127
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -284.0418396, 1149.0468750, -365.4256592, 1476.6849365, -1760.7266846, 1514.4725342
1: -458.4408569, 1271.7687988, -588.9390259, 1634.8612061, -2093.3020020, 1860.7077637
2: -341.8568420, 1464.9821777, -439.4110107, 1882.3299561, -2224.1865234, 1904.3931885
3: -735.3276367, 1310.6555176, -945.5795288, 1685.5665283, -2420.8940430, 2256.2346191
4: -587.7730103, 1372.5423584, -756.9507446, 1762.7757568, -2350.5485840, 2129.4931641

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7353933, upper bound: 1781.7337296
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7336899, upper bound: 1781.7328744
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7342996, upper bound: 1781.7335601
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -283.8130188, 1147.8452148, -366.4806824, 1479.8315430, -1763.6445312, 1514.3259277
1: -458.0265503, 1270.4219971, -590.6148682, 1638.8887939, -2096.9152832, 1861.0368652
2: -341.5471802, 1463.4454346, -440.8607483, 1886.4167480, -2227.9636230, 1904.3061523
3: -734.6736450, 1309.2102051, -948.4612427, 1689.8979492, -2424.5715332, 2257.6711426
4: -587.1547241, 1371.1164551, -759.5183716, 1766.9053955, -2354.0600586, 2130.6340332

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7351453, upper bound: 1781.7332436
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7339353, upper bound: 1781.7332163
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7344764, upper bound: 1781.7338815
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -236.6545868, 959.6105347, -334.2898560, 1350.2215576, -1586.8760986, 1293.9003906
1: -381.4501648, 1062.1062012, -538.7498169, 1496.5661621, -1878.0163574, 1600.8557129
2: -284.1340332, 1223.8709717, -402.4730530, 1721.9534912, -2006.0875244, 1626.3437500
3: -613.5445557, 1091.1414795, -868.3084717, 1539.4041748, -2152.9487305, 1959.4495850
4: -489.2980652, 1144.2718506, -693.0247192, 1611.9007568, -2101.1987305, 1837.2965088

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7334020, upper bound: 1781.7318610
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7296678, upper bound: 1781.7298063
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7337480, upper bound: 1781.7326911
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -236.2603149, 958.0255127, -351.1420593, 1415.4035645, -1651.6635742, 1309.1673584
1: -380.6958313, 1060.3537598, -566.3146362, 1569.4213867, -1950.1171875, 1626.6684570
2: -283.5668945, 1221.8634033, -422.8389893, 1805.1079102, -2088.6748047, 1644.7023926
3: -612.2658691, 1089.3487549, -912.3206177, 1615.7717285, -2228.0375977, 2001.6693115
4: -488.5026245, 1142.2976074, -727.7595825, 1691.1485596, -2179.6511230, 1870.0570068

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7320147, upper bound: 1781.7312794
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7319004, upper bound: 1781.7311836
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -246.5519867, 1000.1772461, -337.4582214, 1363.0609131, -1609.6127930, 1337.6354980
1: -397.2989807, 1106.9346924, -543.8497925, 1510.7229004, -1908.0217285, 1650.7843018
2: -296.0162048, 1275.4351807, -406.2485046, 1738.3247070, -2034.3408203, 1681.6837158
3: -638.7319946, 1138.2556152, -876.4279785, 1554.0594482, -2192.7915039, 2014.6833496
4: -510.0746155, 1193.1069336, -699.5986328, 1627.3126221, -2137.3872070, 1892.7054443

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7339824, upper bound: 1781.7328482
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7343229, upper bound: 1781.7335820
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7343846, upper bound: 1781.7336048
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -245.8666534, 997.3900757, -354.5322876, 1428.8747559, -1674.7413330, 1351.9223633
1: -396.1149292, 1103.8647461, -571.7579346, 1584.4075928, -1980.5224609, 1675.6225586
2: -295.1096802, 1271.9189453, -426.8937683, 1822.2912598, -2117.4008789, 1698.8127441
3: -636.7738647, 1135.0129395, -921.0518799, 1631.2690430, -2268.0429688, 2056.0646973
4: -508.6737671, 1189.7017822, -734.8184204, 1707.3771973, -2216.0510254, 1924.5201416

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7328487, upper bound: 1781.7321869
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7320414, upper bound: 1781.7312850
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -236.6545868, 959.6105347, -404.9357910, 1637.3265381, -1873.9810791, 1364.5463867
1: -381.4501648, 1062.1062012, -650.8153076, 1813.6697998, -2195.1198730, 1712.9215088
2: -284.1340332, 1223.8709717, -486.2390442, 2087.2722168, -2371.4062500, 1710.1099854
3: -613.5445557, 1091.1414795, -1047.7667236, 1866.4033203, -2479.9477539, 2138.9082031
4: -489.2980652, 1144.2718506, -838.6550903, 1952.0727539, -2441.3706055, 1982.9267578

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7334020, upper bound: 1781.7318610
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7324804, upper bound: 1781.7301069
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 25

Time for candidate selection: 6.92 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7338664, upper bound: 1781.7326012
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7334893, upper bound: 1781.7327104
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7282365, upper bound: 1781.7298465
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7331747, upper bound: 1781.7322493
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7329356, upper bound: 1781.7320841
time: 1.17 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -236.2603149, 958.0255127, -419.6219788, 1694.4675293, -1930.7276611, 1377.6474609
1: -380.6958313, 1060.3537598, -674.7062988, 1877.4016113, -2258.0971680, 1735.0600586
2: -283.5668945, 1221.8634033, -503.9333496, 2160.3420410, -2443.9089355, 1725.7966309
3: -612.2658691, 1089.3487549, -1085.9721680, 1933.5828857, -2545.8486328, 2175.3203125
4: -488.5026245, 1142.2976074, -869.0918579, 2021.8012695, -2510.3037109, 2011.3892822

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7335122, upper bound: 1781.7315624
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7297015, upper bound: 1781.7299419
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7339935, upper bound: 1781.7327993
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -246.5519867, 1000.1772461, -406.4460754, 1643.5296631, -1890.0815430, 1406.6230469
1: -397.2989807, 1106.9346924, -653.2752075, 1820.5286865, -2217.8276367, 1760.2097168
2: -296.0162048, 1275.4351807, -488.0657349, 2095.2145996, -2391.2307129, 1763.5009766
3: -638.7319946, 1138.2556152, -1051.7313232, 1873.4188232, -2512.1508789, 2189.9865723
4: -510.0746155, 1193.1069336, -841.8118896, 1959.4713135, -2469.5458984, 2034.9188232

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7339824, upper bound: 1781.7328482
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7343229, upper bound: 1781.7335820
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7343846, upper bound: 1781.7336048
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -245.8666534, 997.3900757, -421.1542969, 1700.5468750, -1946.4134521, 1418.5443115
1: -396.1149292, 1103.8647461, -677.1813354, 1884.2280273, -2280.3422852, 1781.0461426
2: -295.1096802, 1271.9189453, -505.7862854, 2168.1140137, -2463.2236328, 1777.7052002
3: -636.7738647, 1135.0129395, -1090.0067139, 1940.5679932, -2577.3417969, 2225.0192871
4: -508.6737671, 1189.7017822, -872.3142090, 2029.0946045, -2537.7680664, 2062.0158691

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

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
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7345626, upper bound: 1781.7337128
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7346044, upper bound: 1781.7337259
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -270.0743408, 1096.1451416, -345.6982727, 1396.0196533, -1666.0938721, 1441.8433838
1: -435.7109680, 1213.2921143, -557.4037476, 1547.2988281, -1983.0097656, 1770.6958008
2: -325.0308533, 1397.0957031, -416.3791199, 1780.1428223, -2105.1735840, 1813.4747314
3: -699.4786987, 1248.7093506, -897.7531738, 1592.0480957, -2291.5261230, 2146.4624023
4: -559.2321167, 1307.8543701, -716.7529297, 1666.8232422, -2226.0551758, 2024.6072998

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7340362, upper bound: 1781.7334769
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7342628, upper bound: 1781.7341638
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -270.0743408, 1096.1451416, -363.6672058, 1465.8568115, -1735.9310303, 1459.8123779
1: -435.7109680, 1213.2921143, -586.7440186, 1624.9959717, -2060.7065430, 1800.0361328
2: -325.0308533, 1397.0957031, -438.0099487, 1869.3762207, -2194.4069824, 1835.1057129
3: -699.4786987, 1248.7093506, -944.5310669, 1673.1320801, -2372.6103516, 2193.2404785
4: -559.2321167, 1307.8543701, -753.5618286, 1751.4698486, -2310.7009277, 2061.4155273

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7340362, upper bound: 1781.7338975
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7342628, upper bound: 1781.7345342
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -287.6105042, 1163.5479736, -345.6982727, 1396.0196533, -1683.6301270, 1509.2460938
1: -464.1067200, 1287.8089600, -557.4037476, 1547.2988281, -2011.4055176, 1845.2126465
2: -346.1171265, 1483.4461670, -416.3791199, 1780.1428223, -2126.2600098, 1899.8250732
3: -744.4826050, 1327.0776367, -897.7531738, 1592.0480957, -2336.5297852, 2224.8308105
4: -595.1425171, 1389.7296143, -716.7529297, 1666.8232422, -2261.9658203, 2106.4819336

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7340135, upper bound: 1781.7336425
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7345342, upper bound: 1781.7343588
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -287.6105042, 1163.5479736, -363.6682129, 1465.8612061, -1753.4715576, 1527.2161865
1: -464.1067200, 1287.8089600, -586.7456665, 1625.0006104, -2089.1071777, 1874.5546875
2: -346.1171265, 1483.4461670, -438.0113525, 1869.3819580, -2215.4990234, 1921.4573975
3: -744.4826050, 1327.0776367, -944.5338745, 1673.1372070, -2417.6191406, 2271.6115723
4: -595.1425171, 1389.7296143, -753.5640869, 1751.4750977, -2346.6169434, 2143.2929688

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7340135, upper bound: 1781.7339676
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7345342, upper bound: 1781.7345648
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -270.0743408, 1096.1451416, -415.4491882, 1679.8620605, -1949.9362793, 1511.5943604
1: -435.7109680, 1213.2921143, -667.9820557, 1860.7307129, -2296.4416504, 1881.2740479
2: -325.0308533, 1397.0957031, -499.0282898, 2141.3107910, -2466.3415527, 1896.1239014
3: -699.4786987, 1248.7093506, -1074.8863525, 1915.0198975, -2614.4978027, 2323.5957031
4: -559.2321167, 1307.8543701, -860.4401855, 2002.8547363, -2562.0861816, 2168.2941895

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7340362, upper bound: 1781.7334769
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7342628, upper bound: 1781.7341638
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -270.0743408, 1096.1451416, -431.2170410, 1741.7592773, -2011.8336182, 1527.3621826
1: -435.7109680, 1213.2921143, -693.6503296, 1929.4581299, -2365.1691895, 1906.9423828
2: -325.0308533, 1397.0957031, -517.9996338, 2220.4948730, -2545.5253906, 1915.0952148
3: -699.4786987, 1248.7093506, -1115.9855957, 1987.3259277, -2686.8039551, 2364.6948242
4: -559.2321167, 1307.8543701, -893.0153198, 2078.2082520, -2637.4399414, 2200.8691406

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7340362, upper bound: 1781.7336373
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7342628, upper bound: 1781.7343166
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -287.6105042, 1163.5479736, -415.4491882, 1679.8620605, -1967.4724121, 1578.9971924
1: -464.1067200, 1287.8089600, -667.9820557, 1860.7307129, -2324.8371582, 1955.7910156
2: -346.1171265, 1483.4461670, -499.0282898, 2141.3107910, -2487.4279785, 1982.4741211
3: -744.4826050, 1327.0776367, -1074.8863525, 1915.0198975, -2659.5017090, 2401.9638672
4: -595.1425171, 1389.7296143, -860.4401855, 2002.8547363, -2597.9968262, 2250.1691895

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7340135, upper bound: 1781.7336425
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7345342, upper bound: 1781.7343588
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -287.6105042, 1163.5479736, -431.2170410, 1741.7592773, -2029.3697510, 1594.7650146
1: -464.1067200, 1287.8089600, -693.6503296, 1929.4581299, -2393.5646973, 1981.4592285
2: -346.1171265, 1483.4461670, -517.9996338, 2220.4948730, -2566.6115723, 2001.4455566
3: -744.4826050, 1327.0776367, -1115.9855957, 1987.3259277, -2731.8078613, 2443.0632324
4: -595.1425171, 1389.7296143, -893.0153198, 2078.2082520, -2673.3505859, 2282.7441406

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7340135, upper bound: 1781.7337996
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7345342, upper bound: 1781.7345064
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -355.2095337, 1435.9851074, -334.4087524, 1351.2254639, -1706.4350586, 1770.3937988
1: -571.8370972, 1590.2640381, -538.5978394, 1496.1608887, -2067.9978027, 2128.8618164
2: -427.1219788, 1830.6579590, -401.7649536, 1723.1145020, -2150.2363281, 2232.4228516
3: -919.3923950, 1638.5804443, -865.6705933, 1541.4085693, -2460.8010254, 2504.2509766
4: -736.5457764, 1713.3140869, -693.1326904, 1612.5623779, -2349.1079102, 2406.4467773

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7331460, upper bound: 1781.7297069
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7331460, upper bound: 1781.7297069
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -343.6361389, 1390.1774902, -316.2273560, 1282.2316895, -1625.8677979, 1706.4047852
1: -553.4163818, 1539.4211426, -510.3652039, 1418.2913818, -1971.7077637, 2049.7863770
2: -413.3297119, 1772.2858887, -380.3400574, 1635.1579590, -2048.4877930, 2152.6259766
3: -889.5669556, 1585.8253174, -819.3826904, 1461.9344482, -2351.5014648, 2405.2080078
4: -712.7155151, 1658.3372803, -656.0921631, 1530.1651611, -2242.8806152, 2314.4287109

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7262117, upper bound: 1781.7202140
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7181147, upper bound: 1781.7144911
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -367.6542358, 1484.8155518, -334.7695007, 1352.7100830, -1720.3641357, 1819.5849609
1: -592.2290039, 1644.5460205, -539.1025391, 1497.7438965, -2089.9729004, 2183.6481934
2: -442.2187500, 1892.9051514, -402.1510620, 1724.9703369, -2167.1889648, 2295.0559082
3: -951.5325928, 1696.3447266, -866.3568115, 1543.2717285, -2494.8041992, 2562.7011719
4: -762.4922485, 1773.1385498, -693.9865112, 1614.3436279, -2376.8347168, 2467.1245117

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7295264, upper bound: 1781.7318141
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7295264, upper bound: 1781.7330288
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -356.8674927, 1442.2946777, -317.0585327, 1285.6649170, -1642.5324707, 1759.3531494
1: -575.0257568, 1597.2181396, -511.6426086, 1422.0307617, -1997.0565186, 2108.8608398
2: -429.3293762, 1838.7312012, -381.2789917, 1639.5378418, -2068.8669434, 2220.0102539
3: -923.6262817, 1647.2397461, -821.2687988, 1466.0892334, -2389.7155762, 2468.5083008
4: -740.2330322, 1722.0755615, -657.9198608, 1534.3310547, -2274.5639648, 2379.9951172

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7286788, upper bound: 1781.7286788
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7286788, upper bound: 1781.7297733
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -362.8729553, 1466.9240723, -367.1136780, 1484.7327881, -1847.6055908, 1834.0377197
1: -584.3854370, 1624.5229492, -591.1312256, 1644.1737061, -2228.5588379, 2215.6535645
2: -436.4882812, 1869.8391113, -441.6200256, 1892.3029785, -2328.7912598, 2311.4592285
3: -939.0621338, 1674.0444336, -949.1871948, 1695.4190674, -2634.4809570, 2623.2314453
4: -752.3818359, 1750.2965088, -761.6397705, 1771.7617188, -2524.1433105, 2511.9360352

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7342559, upper bound: 1781.7330038
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7342559, upper bound: 1781.7330038
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -351.3864746, 1421.4680176, -350.5257874, 1421.6638184, -1773.0501709, 1771.9937744
1: -566.1113892, 1574.0505371, -565.5172729, 1573.0462646, -2139.1577148, 2139.5678711
2: -422.8002625, 1811.9139404, -422.0599670, 1812.0531006, -2234.8530273, 2233.9738770
3: -909.4606934, 1621.6937256, -906.9831543, 1622.9897461, -2532.4504395, 2528.6767578
4: -728.7283325, 1695.7500000, -727.9076538, 1696.6965332, -2425.4245605, 2423.6574707

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7261758, upper bound: 1781.7220431
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7184918, upper bound: 1781.7167743
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -376.0868530, 1519.0712891, -369.4372559, 1493.9371338, -1870.0239258, 1888.5085449
1: -606.0459595, 1682.1925049, -594.7396240, 1654.3031006, -2260.3491211, 2276.9321289
2: -452.4778442, 1936.4600830, -444.3861084, 1904.0585938, -2356.5354004, 2380.8461914
3: -973.1510010, 1735.1192627, -954.9334106, 1706.2142334, -2679.3652344, 2690.0524902
4: -779.7677002, 1813.9238281, -766.6333008, 1782.8349609, -2562.6025391, 2580.5571289

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7295264, upper bound: 1781.7341066
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7295264, upper bound: 1781.7350258
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -365.2734985, 1476.3994141, -353.3084412, 1432.8666992, -1798.1398926, 1829.7076416
1: -588.7783203, 1634.6789551, -569.8681641, 1585.3579102, -2174.1352539, 2204.5466309
2: -439.5636902, 1882.0828857, -425.3794556, 1826.3464355, -2265.9101562, 2307.4624023
3: -945.1253662, 1685.8437500, -913.8768311, 1636.0717773, -2581.1960449, 2599.7207031
4: -757.4555664, 1762.6726074, -733.9022217, 1710.1395264, -2467.5947266, 2496.5747070

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7286788, upper bound: 1781.7318812
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7286788, upper bound: 1781.7330038
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -403.7276917, 1632.7705078, -317.1852417, 1281.5498047, -1685.2774658, 1949.9558105
1: -648.6849976, 1808.5607910, -510.9473572, 1419.2763672, -2067.9611816, 2319.5080566
2: -484.7058716, 2081.3149414, -381.0093689, 1634.4176025, -2119.1232910, 2462.3242188
3: -1044.2111816, 1861.0443115, -821.5131836, 1460.4865723, -2504.6977539, 2682.5573730
4: -836.0808716, 1946.3917236, -656.9014893, 1528.8073730, -2364.8881836, 2603.2929688

Time for backsubstitution: 1.58 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=2073.053466796875
rel_dist={0: [-1781.7403846768325, 1781.740384676833]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7388007, upper bound: 1781.7388007
time: 0.68 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7388007, upper bound: 1781.7388007
time: 0.55 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.35 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.35
Output dim: 0, lower bound: -1781.7388007, upper bound: 1781.7388007
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.35
Output dim: 0, lower bound: -1781.7388007, upper bound: 1781.7388007

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -292.2078247, 1185.7346191, -395.0211487, 1597.6091309, -1889.8168945, 1580.7556152
1: -470.8653564, 1312.1569824, -635.5115356, 1768.3704834, -2239.2358398, 1947.6683350
2: -351.4121094, 1511.5142822, -474.6130371, 2036.1599121, -2387.5720215, 1986.1271973
3: -755.9409180, 1351.1541748, -1020.9802856, 1822.9670410, -2578.9079590, 2372.1342773
4: -605.2770386, 1414.5281982, -818.1955566, 1906.2989502, -2511.5759277, 2232.7236328

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7379405, upper bound: 1781.7377676
time: 0.52 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7380566, upper bound: 1781.7380566
time: 0.56 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -407.7161560, 1648.8684082, -409.0175171, 1654.4312744, -2062.1474609, 2057.8859863
1: -656.0112915, 1825.3117676, -658.0963745, 1831.3126221, -2487.3234863, 2483.4082031
2: -490.0357056, 2101.4226074, -491.5933533, 2108.5288086, -2598.5644531, 2593.0158691
3: -1053.7666016, 1881.9282227, -1057.0537109, 1888.0509033, -2941.8173828, 2938.9819336
4: -844.8474731, 1967.4259033, -847.5256958, 1973.9653320, -2818.8127441, 2814.9509277

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7379405, upper bound: 1781.7377676
time: 0.53 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7380566, upper bound: 1781.7380566
time: 0.59 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.44 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.44
Output dim: 0, lower bound: -1781.7379405, upper bound: 1781.7377676
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.44
Output dim: 0, lower bound: -1781.7380566, upper bound: 1781.7380566
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.44
Output dim: 0, lower bound: -1781.7379405, upper bound: 1781.7377676
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.44
Output dim: 0, lower bound: -1781.7380566, upper bound: 1781.7380566

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -281.3028870, 1141.6490479, -367.9794922, 1487.5091553, -1768.8120117, 1509.6281738
1: -453.3091736, 1263.3961182, -592.2954102, 1646.8975830, -2100.2067871, 1855.6915283
2: -338.2549438, 1455.3948975, -442.3876953, 1896.2772217, -2234.5314941, 1897.7825928
3: -727.7110596, 1300.9315186, -951.4140015, 1697.8239746, -2425.5351562, 2252.3454590
4: -582.6524048, 1361.9649658, -762.9083252, 1775.0401611, -2357.6926270, 2124.8725586

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7372675, upper bound: 1781.7370165
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7379405, upper bound: 1781.7377676
time: 0.54 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -289.6578979, 1175.1263428, -420.8031616, 1701.1618652, -1990.8198242, 1595.9291992
1: -466.7900391, 1300.5187988, -676.2272949, 1883.9473877, -2350.7373047, 1976.7459717
2: -348.3545227, 1498.1142578, -505.3245544, 2168.4853516, -2516.8398438, 2003.4388428
3: -749.4802856, 1338.9455566, -1087.5737305, 1940.5196533, -2690.0000000, 2426.5192871
4: -600.0190430, 1401.7679443, -871.7164917, 2028.9907227, -2629.0092773, 2273.4843750

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7374317, upper bound: 1781.7373039
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7380566, upper bound: 1781.7380566
time: 0.64 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -391.8676453, 1584.3438721, -380.9550171, 1540.3983154, -1932.2658691, 1965.2988281
1: -630.7034912, 1754.1191406, -613.2868652, 1705.4475098, -2336.1508789, 2367.4060059
2: -471.1496582, 2019.4416504, -458.1799011, 1963.6135254, -2434.7631836, 2477.6215820
3: -1013.0391846, 1808.5925293, -984.9059448, 1758.4722900, -2771.5114746, 2793.4985352
4: -812.4279785, 1890.5158691, -790.1844482, 1838.0255127, -2650.4533691, 2680.7001953

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7358705, upper bound: 1781.7366606
time: 0.51 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7379405, upper bound: 1781.7377676
time: 0.59 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -405.3283691, 1639.3203125, -435.1426086, 1759.9331055, -2165.2614746, 2074.4628906
1: -652.1103516, 1814.6867676, -699.5060425, 1948.9365234, -2601.0468750, 2514.1928711
2: -487.0739746, 2089.1757812, -522.6465454, 2243.4631348, -2730.5368652, 2611.8222656
3: -1047.5103760, 1870.9151611, -1125.1517334, 2007.0686035, -3054.5791016, 2996.0666504
4: -839.7683716, 1955.8560791, -901.5311279, 2098.6118164, -2938.3801270, 2857.3867188

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7356449, upper bound: 1781.7354919
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7355485, upper bound: 1781.7355485
time: 0.57 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.50 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -1781.7372675, upper bound: 1781.7370165
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -1781.7379405, upper bound: 1781.7377676
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -1781.7374317, upper bound: 1781.7373039
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -1781.7380566, upper bound: 1781.7380566
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -1781.7358705, upper bound: 1781.7366606
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -1781.7379405, upper bound: 1781.7377676
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -1781.7356449, upper bound: 1781.7354919
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -1781.7355485, upper bound: 1781.7355485

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -244.9406281, 993.6793213, -352.9029541, 1426.5000000, -1671.4406738, 1346.5816650
1: -394.6773987, 1099.6469727, -567.9536133, 1579.4373779, -1974.1147461, 1667.6005859
2: -293.9783936, 1267.2098389, -424.0415344, 1818.6611328, -2112.6391602, 1691.2513428
3: -634.2202759, 1131.3353271, -912.5075073, 1628.3804932, -2262.6003418, 2043.8426514
4: -506.6963196, 1185.6580811, -731.7893066, 1702.3071289, -2209.0031738, 1917.4473877

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7354637, upper bound: 1781.7344303
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7354637, upper bound: 1781.7343486
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -276.5448303, 1122.6771240, -364.7381592, 1474.4080811, -1750.9528809, 1487.4152832
1: -445.7642517, 1242.7175293, -587.1149292, 1632.5963135, -2078.3601074, 1829.8323975
2: -332.6376038, 1430.8443604, -438.5502319, 1879.3907471, -2212.0278320, 1869.3944092
3: -715.2825317, 1280.0599365, -942.9068604, 1683.4088135, -2398.6904297, 2222.9663086
4: -573.0807495, 1339.5045166, -756.3555908, 1759.6014404, -2332.6821289, 2095.8601074

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7355606, upper bound: 1781.7353842
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7355578, upper bound: 1781.7351203
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -252.8388214, 1025.1055908, -405.3136292, 1638.8458252, -1891.6845703, 1430.4191895
1: -407.4627991, 1134.6143799, -651.1975708, 1814.9608154, -2222.4235840, 1785.8120117
2: -303.5871582, 1307.3388672, -486.4962158, 2089.1840820, -2392.7712402, 1793.8350830
3: -654.8781738, 1166.9957275, -1047.5152588, 1869.3914795, -2524.2695312, 2214.5102539
4: -523.1829224, 1223.0378418, -839.6832275, 1954.5549316, -2477.7375488, 2062.7211914

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7356353, upper bound: 1781.7346911
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7356638, upper bound: 1781.7347784
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -284.4882812, 1154.4042969, -417.8143921, 1689.0924072, -1973.5805664, 1572.2187500
1: -458.6047363, 1277.9477539, -671.4601440, 1870.7839355, -2329.3884277, 1949.4079590
2: -342.2436218, 1471.3449707, -501.7886963, 2152.9440918, -2495.1877441, 1973.1336670
3: -735.9957275, 1316.0233154, -1079.7374268, 1927.1501465, -2663.1455078, 2395.7604980
4: -589.5635376, 1377.1777344, -865.6972046, 2014.6704102, -2604.2338867, 2242.8750000

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7357182, upper bound: 1781.7356449
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7357298, upper bound: 1781.7355485
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -352.4555969, 1424.1561279, -365.4784851, 1477.9515381, -1830.4071045, 1789.6346436
1: -567.4793091, 1576.8776855, -588.2853394, 1636.3353271, -2203.8146973, 2165.1628418
2: -423.3810730, 1815.8482666, -439.3923950, 1884.1972656, -2307.5778809, 2255.2402344
3: -912.0493774, 1624.9379883, -944.9912109, 1687.3599854, -2599.4091797, 2569.9287109
4: -730.3623047, 1699.6166992, -758.2932129, 1763.5316162, -2493.8940430, 2457.9099121

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7345380, upper bound: 1781.7342207
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7346447, upper bound: 1781.7342253
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -385.6802368, 1559.4901123, -377.6306763, 1527.0158691, -1912.6960449, 1937.1208496
1: -620.8372803, 1726.9355469, -607.9850464, 1690.8193359, -2311.6567383, 2334.9201660
2: -463.8261414, 1987.4316406, -454.2472534, 1946.3753662, -2410.2009277, 2441.6789551
3: -996.8547974, 1781.0463867, -976.2064209, 1743.7145996, -2740.5693359, 2757.2529297
4: -799.9069824, 1861.0926514, -783.4692993, 1822.2298584, -2622.1367188, 2644.5620117

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7354364, upper bound: 1781.7353842
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7354416, upper bound: 1781.7351203
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -389.8228149, 1576.1888428, -424.6961365, 1717.4018555, -2107.2246094, 2000.8848877
1: -627.3713989, 1745.1652832, -682.7653198, 1902.0709229, -2529.4416504, 2427.9306641
2: -468.4550171, 2008.8022461, -510.1032410, 2189.2875977, -2657.7426758, 2518.9055176
3: -1008.4271240, 1798.0472412, -1098.5787354, 1957.9539795, -2966.3811035, 2896.6259766
4: -807.2673340, 1880.4361572, -879.6713867, 2047.7561035, -2855.0231934, 2760.1071777

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7356449, upper bound: 1781.7354919
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7356449, upper bound: 1781.7354919
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -403.2672119, 1629.5780029, -429.1786804, 1735.7858887, -2139.0529785, 2058.7565918
1: -649.4585571, 1804.2647705, -689.7868042, 1922.1594238, -2571.6174316, 2494.0507812
2: -484.7882690, 2076.8759766, -515.4361572, 2212.6940918, -2697.4824219, 2592.3117676
3: -1043.2624512, 1860.5947266, -1109.7121582, 1979.2431641, -3022.5053711, 2970.3068848
4: -835.2879028, 1945.4602051, -889.2224731, 2069.7229004, -2905.0102539, 2834.6823730

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7355485, upper bound: 1781.7355485
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7355485, upper bound: 1781.7355485
time: 0.57 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.60 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -1781.7354637, upper bound: 1781.7344303
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -1781.7354637, upper bound: 1781.7343486
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -1781.7355606, upper bound: 1781.7353842
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -1781.7355578, upper bound: 1781.7351203
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -1781.7356353, upper bound: 1781.7346911
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -1781.7356638, upper bound: 1781.7347784
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -1781.7357182, upper bound: 1781.7356449
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -1781.7357298, upper bound: 1781.7355485
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -1781.7345380, upper bound: 1781.7342207
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -1781.7346447, upper bound: 1781.7342253
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -1781.7354364, upper bound: 1781.7353842
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -1781.7354416, upper bound: 1781.7351203
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -1781.7356449, upper bound: 1781.7354919
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -1781.7356449, upper bound: 1781.7354919
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -1781.7355485, upper bound: 1781.7355485
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -1781.7355485, upper bound: 1781.7355485

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -238.0300903, 965.9193726, -338.0680542, 1366.1763916, -1604.2062988, 1303.9874268
1: -383.6456299, 1068.7926025, -544.1514893, 1513.0144043, -1896.6597900, 1612.9440918
2: -285.7463074, 1231.7608643, -406.1799927, 1741.8618164, -2027.6081543, 1637.9409180
3: -616.8056030, 1099.0025635, -875.0838013, 1558.7583008, -2175.5632324, 1974.0861816
4: -492.2194214, 1152.5229492, -700.6473389, 1630.2175293, -2122.4370117, 1853.1702881

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7339579, upper bound: 1781.7329881
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7352460, upper bound: 1781.7342854
time: 0.50 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7353131, upper bound: 1781.7343498
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -236.8402405, 960.8958740, -349.4664917, 1410.1663818, -1647.0062256, 1310.3623047
1: -381.5737000, 1063.3432617, -562.7460938, 1562.4853516, -1944.0590820, 1626.0892334
2: -284.1658020, 1225.4318848, -420.1601562, 1797.7680664, -2081.9335938, 1645.5920410
3: -613.4397583, 1093.3715820, -904.7777710, 1611.6185303, -2225.0581055, 1998.1491699
4: -489.8287354, 1146.5133057, -724.7001953, 1684.3931885, -2174.2219238, 1871.2135010

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7340376, upper bound: 1781.7324467
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7352752, upper bound: 1781.7342239
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7353329, upper bound: 1781.7342655
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -267.5036926, 1086.1865234, -349.6926270, 1413.2130127, -1680.7166748, 1435.8790283
1: -431.3687134, 1202.2186279, -563.0896606, 1565.1918945, -1996.5605469, 1765.3083496
2: -321.8128052, 1384.2905273, -420.4489441, 1801.3479004, -2123.1606445, 1804.7395020
3: -692.3338623, 1237.7574463, -904.9539795, 1612.8666992, -2305.2006836, 2142.7114258
4: -554.0227051, 1295.9689941, -724.7152710, 1686.4971924, -2240.5200195, 2020.6840820

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7344726, upper bound: 1781.7341093
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7327640, upper bound: 1781.7325748
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7335396, upper bound: 1781.7331373
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -269.9805908, 1095.9699707, -362.6118164, 1463.8450928, -1733.8256836, 1458.5817871
1: -435.0938721, 1213.1685791, -584.3483276, 1621.4357910, -2056.5295410, 1797.5168457
2: -324.7176514, 1396.8776855, -436.1636963, 1866.0058594, -2190.7233887, 1833.0413818
3: -698.5385132, 1249.2406006, -938.4409790, 1672.3641357, -2370.9018555, 2187.6813965
4: -559.5317383, 1307.6717529, -751.6152954, 1748.2431641, -2307.7749023, 2059.2861328

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7344067, upper bound: 1781.7331933
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7328142, upper bound: 1781.7324731
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7335723, upper bound: 1781.7330456
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -245.4520874, 995.3872681, -389.0518188, 1572.6616211, -1818.1137695, 1384.4390869
1: -395.6525879, 1101.5933838, -625.1068726, 1742.0849609, -2137.7375488, 1726.7001953
2: -294.7839661, 1269.4085693, -466.9843140, 2004.8979492, -2299.6811523, 1736.3928223
3: -636.2282104, 1132.5853271, -1006.2973633, 1793.1868896, -2429.4150391, 2138.8828125
4: -507.7352295, 1187.6394043, -805.7039795, 1875.5488281, -2383.2832031, 1993.3432617

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7354827, upper bound: 1781.7340262
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7354346, upper bound: 1781.7345440
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7355666, upper bound: 1781.7346084
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -244.9916382, 993.6305542, -403.5371094, 1628.1168213, -1873.1082764, 1397.1677246
1: -394.7560425, 1099.6500244, -648.5841675, 1804.5528564, -2199.3088379, 1748.2341309
2: -294.0639954, 1267.2136230, -484.5338745, 2075.6672363, -2369.7312012, 1751.7474365
3: -634.6427002, 1130.5202637, -1044.1988525, 1859.0816650, -2493.7243652, 2174.7189941
4: -506.7873230, 1185.3546143, -835.9475098, 1943.4398193, -2450.2270508, 2021.3021240

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7355024, upper bound: 1781.7346704
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7355969, upper bound: 1781.7347109
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -274.8685913, 1115.6485596, -401.5513611, 1622.8474121, -1897.7159424, 1517.1995850
1: -443.3095093, 1234.9638672, -645.4315796, 1797.8508301, -2241.1601562, 1880.3955078
2: -330.7430420, 1421.9334717, -482.2560730, 2068.5122070, -2399.2551270, 1904.1895752
3: -711.6098633, 1271.2246094, -1038.4965820, 1850.8900146, -2562.4992676, 2309.7211914
4: -569.3334351, 1330.9384766, -831.6562500, 1935.5733643, -2504.9067383, 2162.5947266

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7329053, upper bound: 1781.7328051
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7335992, upper bound: 1781.7333698
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -278.3551331, 1129.5888672, -417.3604431, 1684.5561523, -1962.9111328, 1546.9492188
1: -448.5566101, 1250.6295166, -671.2744141, 1866.6159668, -2315.1723633, 1921.9038086
2: -334.8039551, 1439.7353516, -501.3760986, 2147.4189453, -2482.2229004, 1941.1113281
3: -720.2172852, 1287.5239258, -1079.9530029, 1923.1036377, -2643.3208008, 2367.4770508
4: -576.8609009, 1347.5163574, -864.4461060, 2010.6398926, -2587.5007324, 2211.9624023

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7329956, upper bound: 1781.7329483
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7336928, upper bound: 1781.7334814
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -344.6027832, 1392.1801758, -350.6637268, 1417.8872070, -1762.4897461, 1742.8438721
1: -554.9009399, 1541.5916748, -564.5357056, 1570.1243896, -2125.0249023, 2106.1267090
2: -413.9527588, 1775.1628418, -421.5807190, 1807.7009277, -2221.6535645, 2196.7436523
3: -892.2717896, 1587.9025879, -907.6253662, 1617.8625488, -2510.1342773, 2495.5275879
4: -713.8822632, 1661.4746094, -727.2080078, 1691.7100830, -2405.5922852, 2388.6821289

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7337153, upper bound: 1781.7339644
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7337153, upper bound: 1781.7342207
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -345.6846008, 1396.6223145, -362.7101440, 1464.6636963, -1810.3482666, 1759.3322754
1: -556.5036621, 1546.3919678, -584.0916138, 1622.4118652, -2178.9155273, 2130.4836426
2: -415.1544189, 1780.7646484, -436.2004089, 1867.2789307, -2282.4333496, 2216.9650879
3: -894.5841675, 1593.2470703, -938.7338867, 1673.6539307, -2568.2380371, 2531.9809570
4: -716.2824097, 1666.7915039, -752.3766479, 1749.2519531, -2465.5339355, 2419.1679688

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7338300, upper bound: 1781.7339598
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7338300, upper bound: 1781.7342253
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -376.0205383, 1520.2274170, -362.8211060, 1466.9604492, -1842.9809570, 1883.0483398
1: -605.4094238, 1683.6782227, -584.3242188, 1624.5472412, -2229.9560547, 2268.0024414
2: -452.2207642, 1937.4196777, -436.4542847, 1869.7814941, -2322.0017090, 2373.8740234
3: -972.4738159, 1735.6385498, -938.8062134, 1674.2589111, -2646.7326660, 2674.4438477
4: -779.6084595, 1814.2006836, -752.3430786, 1750.4244385, -2530.0329590, 2566.5434570

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7349940, upper bound: 1781.7352859
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7349940, upper bound: 1781.7353842
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -380.2702942, 1537.2949219, -376.1757812, 1519.3750000, -1899.6452637, 1913.4707031
1: -612.0079346, 1702.4069824, -606.1272583, 1682.5148926, -2294.5227051, 2308.5341797
2: -457.2718811, 1959.1921387, -452.5650940, 1936.7901611, -2394.0620117, 2411.7570801
3: -982.9696045, 1755.5706787, -973.1119995, 1735.6376953, -2718.6074219, 2728.6826172
4: -788.7259521, 1834.6688232, -779.9248047, 1814.3836670, -2603.1088867, 2614.5937500

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7350196, upper bound: 1781.7350196
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7350196, upper bound: 1781.7351203
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -389.8228149, 1576.1888428, -352.7457886, 1424.4519043, -1814.2746582, 1928.9344482
1: -627.3713989, 1745.1652832, -568.6373901, 1578.6998291, -2206.0705566, 2313.8024902
2: -468.4550171, 2008.8022461, -424.7725220, 1816.5292969, -2284.9841309, 2433.5747070
3: -1008.4271240, 1798.0472412, -915.7263184, 1624.5528564, -2632.9794922, 2713.7734375
4: -807.2673340, 1880.4361572, -731.3771362, 1700.6975098, -2507.9648438, 2611.8127441

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7337936, upper bound: 1781.7337172
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7341542, upper bound: 1781.7348660
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -389.8228149, 1576.1888428, -422.9716492, 1710.3659668, -2100.1882324, 1999.1602783
1: -627.3713989, 1745.1652832, -679.9811401, 1894.3444824, -2521.7153320, 2425.1462402
2: -468.4550171, 2008.8022461, -508.0187988, 2180.2695312, -2648.7246094, 2516.8210449
3: -1008.4271240, 1798.0472412, -1094.0999756, 1949.9609375, -2958.3879395, 2892.1472168
4: -807.2673340, 1880.4361572, -876.0777588, 2039.2756348, -2846.5419922, 2756.5131836

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7337936, upper bound: 1781.7337172
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7341542, upper bound: 1781.7348660
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -403.2672119, 1629.5780029, -354.8948975, 1433.0891113, -1836.3562012, 1984.4729004
1: -649.4585571, 1804.2647705, -571.9155273, 1588.2955322, -2237.7541504, 2376.1794434
2: -484.7882690, 2076.8759766, -427.2713623, 1827.5274658, -2312.3156738, 2504.1464844
3: -1043.2624512, 1860.5947266, -921.0250854, 1634.6455078, -2677.9079590, 2781.6196289
4: -835.2879028, 1945.4602051, -735.9913330, 1710.8775635, -2546.1650391, 2681.4504395

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7347529, upper bound: 1781.7345875
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7355485, upper bound: 1781.7355485
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -403.2672119, 1629.5780029, -427.5389404, 1729.1149902, -2132.3820801, 2057.1169434
1: -649.4585571, 1804.2647705, -687.1311646, 1914.8237305, -2564.2819824, 2491.3957520
2: -484.7882690, 2076.8759766, -513.4505615, 2204.1401367, -2688.9284668, 2590.3261719
3: -1043.2624512, 1860.5947266, -1105.4436035, 1971.6749268, -3014.9375000, 2966.0383301
4: -835.2879028, 1945.4602051, -885.8012085, 2061.6823730, -2896.9697266, 2831.2609863

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7347529, upper bound: 1781.7345875
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7355485, upper bound: 1781.7355485
time: 0.56 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.60 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7352460, upper bound: 1781.7342854
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7353131, upper bound: 1781.7343498
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7352752, upper bound: 1781.7342239
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7353329, upper bound: 1781.7342655
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7327640, upper bound: 1781.7325748
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7335396, upper bound: 1781.7331373
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7328142, upper bound: 1781.7324731
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7335723, upper bound: 1781.7330456
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7354346, upper bound: 1781.7345440
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7355666, upper bound: 1781.7346084
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7355024, upper bound: 1781.7346704
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7355969, upper bound: 1781.7347109
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7329053, upper bound: 1781.7328051
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7335992, upper bound: 1781.7333698
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7329956, upper bound: 1781.7329483
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7336928, upper bound: 1781.7334814
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7337153, upper bound: 1781.7339644
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7337153, upper bound: 1781.7342207
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7338300, upper bound: 1781.7339598
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7338300, upper bound: 1781.7342253
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7349940, upper bound: 1781.7352859
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7349940, upper bound: 1781.7353842
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7350196, upper bound: 1781.7350196
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7350196, upper bound: 1781.7351203
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7337936, upper bound: 1781.7337172
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7341542, upper bound: 1781.7348660
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7337936, upper bound: 1781.7337172
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7341542, upper bound: 1781.7348660
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7347529, upper bound: 1781.7345875
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7355485, upper bound: 1781.7355485
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7347529, upper bound: 1781.7345875
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1781.7355485, upper bound: 1781.7355485

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -234.3851318, 951.7252808, -335.6616211, 1356.7659912, -1591.1510010, 1287.3869629
1: -377.8174438, 1052.7225342, -540.3032837, 1502.5241699, -1880.3415527, 1593.0256348
2: -281.3193054, 1213.5948486, -403.2930908, 1729.8300781, -2011.1494141, 1616.8876953
3: -607.2202759, 1082.6646729, -868.8658447, 1547.8524170, -2155.0722656, 1951.5303955
4: -484.4959717, 1135.6054688, -695.6525879, 1618.8135986, -2103.3095703, 1831.2579346

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7345072, upper bound: 1781.7335560
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7347392, upper bound: 1781.7333570
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7347392, upper bound: 1781.7342854
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -233.2238312, 946.9755859, -336.1672668, 1358.6165771, -1591.8403320, 1283.1428223
1: -375.9743347, 1047.7333984, -541.1017456, 1504.6708984, -1880.6452637, 1588.8350830
2: -280.0008850, 1207.5147705, -403.8886414, 1732.1741943, -2012.1750488, 1611.4033203
3: -604.5816040, 1076.7183838, -870.2106934, 1549.9935303, -2154.5747070, 1946.9289551
4: -482.0498657, 1129.5614014, -696.6466064, 1621.0407715, -2103.0903320, 1826.2078857

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7345976, upper bound: 1781.7335823
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7347460, upper bound: 1781.7333749
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7347460, upper bound: 1781.7343498
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -233.2722778, 947.1171265, -346.7089844, 1399.3077393, -1632.5799561, 1293.8261719
1: -375.8612366, 1047.7353516, -558.3236694, 1550.4067383, -1926.2678223, 1606.0589600
2: -279.8207703, 1207.8197021, -416.8505859, 1783.9127197, -2063.7329102, 1624.6702881
3: -604.0257568, 1077.4395752, -897.6555786, 1599.0260010, -2203.0512695, 1975.0949707
4: -482.2726440, 1129.9818115, -718.9714966, 1671.2579346, -2153.5305176, 1848.9532471

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7347442, upper bound: 1781.7337178
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7349778, upper bound: 1781.7339652
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7349778, upper bound: 1781.7342239
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -232.1515045, 942.6974487, -347.4033508, 1401.9353027, -1634.0867920, 1290.1007080
1: -374.1054993, 1043.0482178, -559.4693604, 1553.3692627, -1927.4747314, 1602.5175781
2: -278.5664673, 1202.1804199, -417.6791687, 1787.2619629, -2065.8281250, 1619.8594971
3: -601.5136108, 1071.8288574, -899.5082397, 1602.0189209, -2203.5324707, 1971.3371582
4: -479.9260864, 1124.3051758, -720.3522949, 1674.4511719, -2154.3771973, 1844.6573486

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7347443, upper bound: 1781.7336606
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7349778, upper bound: 1781.7339995
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7349778, upper bound: 1781.7342655
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -255.4222260, 1036.7119141, -345.2349243, 1394.8699951, -1650.2921143, 1381.9467773
1: -411.8739929, 1147.7011719, -555.8942871, 1544.9923096, -1956.8663330, 1703.5952148
2: -307.3041382, 1321.4265137, -415.0361023, 1777.9428711, -2085.2470703, 1736.4626465
3: -661.4708862, 1180.7807617, -893.4753418, 1591.8901367, -2253.3605957, 2074.2561035
4: -528.8874512, 1236.3277588, -715.3259277, 1664.5527344, -2193.4399414, 1951.6536865

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7305341, upper bound: 1781.7306223
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7315195, upper bound: 1781.7309878
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -264.9081116, 1075.8122559, -347.1640015, 1403.0413818, -1667.9494629, 1422.9763184
1: -427.1638794, 1190.7789307, -558.9822998, 1553.9436035, -1981.1072998, 1749.7609863
2: -318.6824036, 1371.0861816, -417.4287109, 1788.4193115, -2107.1018066, 1788.5148926
3: -685.6380615, 1225.8148193, -898.4351807, 1601.1505127, -2286.7880859, 2124.2495117
4: -548.6725464, 1283.4603271, -719.5412598, 1674.2379150, -2222.9101562, 2003.0015869

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7313201, upper bound: 1781.7312051
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7329220, upper bound: 1781.7316930
time: 0.50 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -258.1014099, 1047.0588379, -357.2008972, 1441.6677246, -1699.7687988, 1404.2597656
1: -415.8366699, 1159.3442383, -575.6007690, 1596.9765625, -2012.8132324, 1734.9450684
2: -310.4244995, 1334.7348633, -429.5877075, 1837.7790527, -2148.2036133, 1764.3225098
3: -668.1187744, 1193.0040283, -924.4891357, 1646.9244385, -2315.0432129, 2117.4931641
4: -534.8150635, 1248.7536621, -740.2149658, 1721.6798096, -2256.4948730, 1988.9686279

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7316747, upper bound: 1781.7316116
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7316747, upper bound: 1781.7324731
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -267.4470825, 1085.6452637, -359.5894470, 1451.2686768, -1718.7158203, 1445.2347412
1: -431.0042114, 1201.8225098, -579.4124756, 1607.7081299, -2038.7124023, 1781.2349854
2: -321.6649780, 1383.7371826, -432.5228882, 1850.0310059, -2171.6960449, 1816.2600098
3: -692.0306396, 1237.4758301, -930.6693115, 1657.9464111, -2349.9770508, 2168.1445312
4: -554.2980347, 1295.3215332, -745.3424683, 1733.1464844, -2287.4443359, 2040.6640625

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7324254, upper bound: 1781.7321977
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7324254, upper bound: 1781.7330456
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -242.0860901, 982.4599609, -386.7203674, 1563.4177246, -1805.5037842, 1369.1801758
1: -390.2928772, 1086.8093262, -621.4240112, 1731.8197021, -2122.1123047, 1708.2333984
2: -290.7363281, 1252.7786865, -464.2030029, 1993.1317139, -2283.8679199, 1716.9815674
3: -627.3269043, 1117.8906250, -1000.3646851, 1782.4868164, -2409.8134766, 2118.2551270
4: -500.6967468, 1172.3365479, -800.8699951, 1864.4046631, -2365.1010742, 1973.2062988

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7354346, upper bound: 1781.7345440
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7354346, upper bound: 1781.7345440
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -241.3955536, 979.4365234, -387.1500854, 1565.1511230, -1806.5466309, 1366.5865479
1: -389.1650696, 1083.8477783, -622.0375977, 1733.7742920, -2122.9394531, 1705.8853760
2: -289.9113464, 1248.9672852, -464.6855164, 1995.2570801, -2285.1684570, 1713.6525879
3: -625.8712158, 1113.7093506, -1001.3485718, 1784.4543457, -2410.3256836, 2115.0578613
4: -499.0629578, 1168.2543945, -801.6976318, 1866.4063721, -2365.4692383, 1969.9520264

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7355666, upper bound: 1781.7346084
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7355666, upper bound: 1781.7346084
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -241.5239716, 980.1023560, -400.7866211, 1617.1589355, -1858.6828613, 1380.8889160
1: -389.2373352, 1084.3793945, -644.2260132, 1792.4262695, -2181.6633301, 1728.6054688
2: -289.8936462, 1249.8817139, -481.2501831, 2061.7416992, -2351.6352539, 1731.1317139
3: -625.5499878, 1115.0968018, -1037.2149658, 1846.4166260, -2471.9665527, 2152.3117676
4: -499.5270691, 1169.2786865, -830.2485962, 1930.2589111, -2429.7858887, 1999.5270996

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7355024, upper bound: 1781.7346704
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7355024, upper bound: 1781.7346704
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -241.4155731, 979.9392090, -401.5150452, 1620.1085205, -1861.5240479, 1381.4542236
1: -388.9958496, 1084.3519287, -645.3496704, 1795.6813965, -2184.6772461, 1729.7016602
2: -289.7564392, 1249.6614990, -482.0989075, 2065.4252930, -2355.1816406, 1731.7603760
3: -625.4493408, 1114.2270508, -1038.9765625, 1849.7249756, -2475.1743164, 2153.2028809
4: -499.1619263, 1168.5201416, -831.6947021, 1933.7160645, -2432.8779297, 2000.2145996

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7355969, upper bound: 1781.7347109
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7355969, upper bound: 1781.7347109
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -262.2507019, 1064.5562744, -397.2702637, 1605.4410400, -1867.6916504, 1461.8265381
1: -422.8553162, 1178.6252441, -638.4699097, 1778.6079102, -2201.4631348, 1817.0947266
2: -315.5536499, 1356.8114014, -477.0830688, 2046.2770996, -2361.8303223, 1833.8945312
3: -679.1784668, 1212.5358887, -1027.3477783, 1830.9937744, -2510.1723633, 2239.8837891
4: -543.0549927, 1269.4216309, -822.7223511, 1914.6840820, -2457.7390137, 2092.1435547

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7329053, upper bound: 1781.7328051
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7329053, upper bound: 1781.7328051
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -271.3920898, 1101.5115967, -399.0724487, 1612.8256836, -1884.2176514, 1500.5837402
1: -437.6673889, 1219.4139404, -641.4169922, 1786.7858887, -2224.4531250, 1860.8308105
2: -326.5475159, 1403.9433594, -479.2839661, 2055.7631836, -2382.3103027, 1883.2272949
3: -702.6350708, 1255.0997314, -1032.0860596, 1839.3797607, -2542.0148926, 2287.1850586
4: -562.2122192, 1313.9533691, -826.5512085, 1923.4959717, -2485.7082520, 2140.5041504

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7335992, upper bound: 1781.7333698
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7335992, upper bound: 1781.7333698
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -265.8637390, 1079.1474609, -412.3116455, 1664.0356445, -1929.8992920, 1491.4591064
1: -428.2418823, 1194.7651367, -663.0775757, 1843.9139404, -2272.1552734, 1857.8427734
2: -319.7312012, 1375.3366699, -495.2739258, 2121.2768555, -2441.0075684, 1870.6105957
3: -688.0132446, 1229.3804932, -1066.8426514, 1899.5733643, -2587.5866699, 2296.2231445
4: -550.8192139, 1286.7707520, -853.9013672, 1986.0201416, -2536.8393555, 2140.6721191

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7329956, upper bound: 1781.7329483
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7329956, upper bound: 1781.7329483
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -274.6970215, 1114.5367432, -413.9167480, 1670.2054443, -1944.9024658, 1528.4533691
1: -442.6765137, 1234.0362549, -665.6918945, 1850.9552002, -2293.6318359, 1899.7281494
2: -330.4278870, 1420.6380615, -497.2329407, 2129.2016602, -2459.6291504, 1917.8709717
3: -710.8623657, 1270.3623047, -1071.1469727, 1906.7117920, -2617.5742188, 2341.5092773
4: -569.3660278, 1329.5314941, -857.3081665, 1993.4460449, -2562.8120117, 2186.8395996

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7336928, upper bound: 1781.7334814
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7336928, upper bound: 1781.7334814
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -332.5947571, 1343.4810791, -350.6637268, 1417.8872070, -1750.4818115, 1694.1447754
1: -535.7384033, 1487.7934570, -564.5357056, 1570.1243896, -2105.8627930, 2052.3288574
2: -399.6703796, 1713.2733154, -421.5807190, 1807.7009277, -2207.3708496, 2134.8540039
3: -861.3531494, 1532.4610596, -907.6253662, 1617.8625488, -2479.2158203, 2440.0861816
4: -689.3419189, 1603.3631592, -727.2080078, 1691.7100830, -2381.0520020, 2330.5712891

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 0
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 1

Time for candidate selection: 5.42 seconds

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7337153, upper bound: 1781.7339618
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7336722, upper bound: 1781.7336089
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -385.5872192, 1558.0108643, -350.6637268, 1417.8872070, -1803.4743652, 1908.6745605
1: -620.0123291, 1725.8679199, -564.5357056, 1570.1243896, -2190.1364746, 2290.4035645
2: -462.6831665, 1986.6916504, -421.5807190, 1807.7009277, -2270.3830566, 2408.2722168
3: -998.5879517, 1775.5187988, -907.6253662, 1617.8625488, -2616.4504395, 2683.1433105
4: -798.1658936, 1858.0738525, -727.2080078, 1691.7100830, -2489.8759766, 2585.2817383

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 0
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 1

Time for candidate selection: 5.44 seconds

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7337153, upper bound: 1781.7342097
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7336722, upper bound: 1781.7341085
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -333.6453552, 1347.7647705, -362.7101440, 1464.6636963, -1798.3089600, 1710.4747314
1: -537.2854004, 1492.4266357, -584.0916138, 1622.4118652, -2159.6972656, 2076.5180664
2: -400.8270874, 1718.6882324, -436.2004089, 1867.2789307, -2268.1059570, 2154.8886719
3: -863.6071167, 1537.6422119, -938.7338867, 1673.6539307, -2537.2609863, 2476.3759766
4: -691.6823120, 1608.5017090, -752.3766479, 1749.2519531, -2440.9340820, 2360.8781738

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7327595, upper bound: 1781.7327595
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7327595, upper bound: 1781.7339598
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -386.8925781, 1563.6878662, -362.7101440, 1464.6636963, -1851.5561523, 1926.3978271
1: -621.8839111, 1731.9256592, -584.0916138, 1622.4118652, -2244.2958984, 2316.0166016
2: -464.1219482, 1993.8768311, -436.2004089, 1867.2789307, -2331.4008789, 2430.0771484
3: -1001.3433228, 1782.0936279, -938.7338867, 1673.6539307, -2674.9973145, 2720.8273926
4: -800.9919434, 1864.7166748, -752.3766479, 1749.2519531, -2550.2438965, 2617.0932617

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7327595, upper bound: 1781.7329451
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7327595, upper bound: 1781.7342253
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -364.4357910, 1473.4122314, -362.8211060, 1466.9604492, -1831.3959961, 1836.2333984
1: -586.9095459, 1631.8986816, -584.3242188, 1624.5472412, -2211.4562988, 2216.2229004
2: -438.4550171, 1877.9049072, -436.4542847, 1869.7814941, -2308.2363281, 2314.3591309
3: -942.6354980, 1682.3880615, -938.8062134, 1674.2589111, -2616.8945312, 2621.1940918
4: -755.9830933, 1758.3249512, -752.3430786, 1750.4244385, -2506.4074707, 2510.6679688

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7332964, upper bound: 1781.7334520
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7343067, upper bound: 1781.7338009
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -417.1489258, 1687.0349121, -362.8211060, 1466.9604492, -1884.1092529, 2049.8559570
1: -670.7308350, 1868.8142090, -584.3242188, 1624.5472412, -2295.2780762, 2453.1384277
2: -501.1305237, 2150.3051758, -436.4542847, 1869.7814941, -2370.9121094, 2586.7592773
3: -1078.9552002, 1924.0225830, -938.8062134, 1674.2589111, -2753.2141113, 2862.8281250
4: -864.3165894, 2011.6191406, -752.3430786, 1750.4244385, -2614.7409668, 2763.9621582

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7332964, upper bound: 1781.7335433
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7343067, upper bound: 1781.7339022
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -368.4826050, 1489.6018066, -376.1757812, 1519.3750000, -1887.8576660, 1865.7775879
1: -593.1979370, 1649.6875000, -606.1272583, 1682.5148926, -2275.7126465, 2255.8144531
2: -443.2740784, 1898.5726318, -452.5650940, 1936.7901611, -2380.0642090, 2351.1369629
3: -952.6441040, 1701.3698730, -973.1119995, 1735.6376953, -2688.2817383, 2674.4814453
4: -764.7128906, 1777.7862549, -779.9248047, 1814.3836670, -2579.0959473, 2557.7109375

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7339598, upper bound: 1781.7338300
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7339598, upper bound: 1781.7350104
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -421.8559875, 1706.2764893, -376.1757812, 1519.3750000, -1941.2309570, 2082.4521484
1: -678.0493774, 1889.9025879, -606.1272583, 1682.5148926, -2360.5642090, 2496.0295410
2: -506.7210083, 2174.7209473, -452.5650940, 1936.7901611, -2443.5107422, 2627.2856445
3: -1090.5709229, 1946.3350830, -973.1119995, 1735.6376953, -2826.2082520, 2919.4470215
4: -874.3435059, 2034.5633545, -779.9248047, 1814.3836670, -2688.7265625, 2814.4882812

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7339598, upper bound: 1781.7338818
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7339598, upper bound: 1781.7350928
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -361.6215515, 1467.4035645, -336.4664612, 1358.0230713, -1719.6446533, 1803.8698730
1: -582.8719482, 1621.9537354, -542.8180542, 1505.0275879, -2087.8994141, 2164.7717285
2: -434.7359009, 1871.0274658, -405.3623352, 1731.9516602, -2166.6875000, 2276.3898926
3: -935.9188232, 1670.0820312, -874.5736694, 1548.2774658, -2484.1962891, 2544.6552734
4: -749.0076294, 1749.5283203, -697.3135376, 1621.8194580, -2370.8271484, 2446.8417969

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7337907, upper bound: 1781.7332026
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7337907, upper bound: 1781.7337172
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -384.7184753, 1555.5555420, -350.1593933, 1414.0527344, -1798.7709961, 1905.7149658
1: -619.0303345, 1722.3521729, -564.4294434, 1567.1804199, -2186.2104492, 2286.7814941
2: -462.2138977, 1982.4936523, -421.6097412, 1803.2382812, -2265.4521484, 2404.1035156
3: -995.2296753, 1774.3940430, -909.0400391, 1612.5762939, -2607.8056641, 2683.4340820
4: -796.5808716, 1855.7541504, -725.9486694, 1688.1986084, -2484.7795410, 2581.7028809

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7332198, upper bound: 1781.7351933
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7341542, upper bound: 1781.7351169
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -361.6215515, 1467.4035645, -406.3024597, 1643.1409912, -2004.7625732, 1873.7059326
1: -582.8719482, 1621.9537354, -653.7130737, 1819.7322998, -2402.6042480, 2275.6667480
2: -434.7359009, 1871.0274658, -488.1813049, 2094.7141113, -2529.4499512, 2359.2087402
3: -935.9188232, 1670.0820312, -1052.0288086, 1872.7624512, -2808.6811523, 2722.1103516
4: -749.0076294, 1749.5283203, -841.3508301, 1959.2219238, -2708.2294922, 2590.8789062

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7337907, upper bound: 1781.7332026
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7337907, upper bound: 1781.7337172
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -384.7184753, 1555.5555420, -420.6557312, 1700.9936523, -2085.7121582, 1976.2111816
1: -619.0303345, 1722.3521729, -676.2158813, 1883.9669189, -2502.9973145, 2398.5681152
2: -462.2138977, 1982.4936523, -505.1895447, 2168.3237305, -2630.5375977, 2487.6831055
3: -995.2296753, 1774.3940430, -1088.1234131, 1939.1958008, -2934.4255371, 2862.5175781
4: -796.5808716, 1855.7541504, -871.2365723, 2028.0688477, -2824.6496582, 2726.9907227

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7331658, upper bound: 1781.7345790
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7341542, upper bound: 1781.7348660
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -361.2375793, 1459.0612793, -340.1502075, 1373.8931885, -1735.1307373, 1799.2114258
1: -581.4947510, 1615.4312744, -548.0383301, 1522.6284180, -2104.1230469, 2163.4694824
2: -433.7883301, 1859.8598633, -409.2392883, 1752.0422363, -2185.8303223, 2269.0988770
3: -935.0210571, 1665.6953125, -882.8147583, 1567.0788574, -2502.0998535, 2548.5100098
4: -747.8795166, 1742.6662598, -705.3482056, 1640.2290039, -2388.1083984, 2448.0139160

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7345950, upper bound: 1781.7343827
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7345950, upper bound: 1781.7345880
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -398.3144226, 1609.3471680, -352.4248352, 1423.1391602, -1821.4536133, 1961.7719727
1: -641.3616333, 1781.9896240, -567.9444580, 1577.4937744, -2218.8552246, 2349.9340820
2: -478.7740173, 2050.8862305, -424.3644714, 1814.6485596, -2293.4223633, 2475.2507324
3: -1030.0928955, 1838.2156982, -914.5147095, 1623.8873291, -2653.9799805, 2752.7302246
4: -824.9984741, 1921.6604004, -731.0996094, 1699.1892090, -2524.1867676, 2652.7600098

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7354919, upper bound: 1781.7354679
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7354919, upper bound: 1781.7355245
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -361.2375793, 1459.0612793, -412.3034363, 1667.6881104, -2028.9256592, 1871.3647461
1: -581.4947510, 1615.4312744, -662.5084229, 1846.8378906, -2428.3325195, 2277.9396973
2: -433.7883301, 1859.8598633, -494.9385071, 2125.9934082, -2559.7817383, 2354.7976074
3: -935.0210571, 1665.6953125, -1066.1412354, 1901.7476807, -2836.7687988, 2731.8364258
4: -747.8795166, 1742.6662598, -854.3181763, 1988.5122070, -2736.3908691, 2596.9843750

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7345950, upper bound: 1781.7343827
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7345950, upper bound: 1781.7345875
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -398.3144226, 1609.3471680, -424.4750977, 1716.7645264, -2115.0788574, 2033.8220215
1: -641.3616333, 1781.9896240, -682.2295532, 1901.3509521, -2542.7126465, 2464.2187500
2: -478.7740173, 2050.8862305, -509.8237000, 2188.2355957, -2667.0095215, 2560.7099609
3: -1030.0928955, 1838.2156982, -1097.4216309, 1957.9818115, -2988.0747070, 2935.6369629
4: -824.9984741, 1921.6604004, -879.6234741, 2047.0275879, -2872.0253906, 2801.2839355

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7354919, upper bound: 1781.7354679
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7354919, upper bound: 1781.7355243
time: 0.61 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.46 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7347392, upper bound: 1781.7333570
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7347392, upper bound: 1781.7342854
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7347460, upper bound: 1781.7333749
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7347460, upper bound: 1781.7343498
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7349778, upper bound: 1781.7339652
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7349778, upper bound: 1781.7342239
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7349778, upper bound: 1781.7339995
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7349778, upper bound: 1781.7342655
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7305341, upper bound: 1781.7306223
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7315195, upper bound: 1781.7309878
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7313201, upper bound: 1781.7312051
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7329220, upper bound: 1781.7316930
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7316747, upper bound: 1781.7316116
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7316747, upper bound: 1781.7324731
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7324254, upper bound: 1781.7321977
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7324254, upper bound: 1781.7330456
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7354346, upper bound: 1781.7345440
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7354346, upper bound: 1781.7345440
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7355666, upper bound: 1781.7346084
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7355666, upper bound: 1781.7346084
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7355024, upper bound: 1781.7346704
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7355024, upper bound: 1781.7346704
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7355969, upper bound: 1781.7347109
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7355969, upper bound: 1781.7347109
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7329053, upper bound: 1781.7328051
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7329053, upper bound: 1781.7328051
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7335992, upper bound: 1781.7333698
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7335992, upper bound: 1781.7333698
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7329956, upper bound: 1781.7329483
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7329956, upper bound: 1781.7329483
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7336928, upper bound: 1781.7334814
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7336928, upper bound: 1781.7334814
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7337153, upper bound: 1781.7339618
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7336722, upper bound: 1781.7336089
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7337153, upper bound: 1781.7342097
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7336722, upper bound: 1781.7341085
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7327595, upper bound: 1781.7327595
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7327595, upper bound: 1781.7339598
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7327595, upper bound: 1781.7329451
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7327595, upper bound: 1781.7342253
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7332964, upper bound: 1781.7334520
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7343067, upper bound: 1781.7338009
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7332964, upper bound: 1781.7335433
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7343067, upper bound: 1781.7339022
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7339598, upper bound: 1781.7338300
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7339598, upper bound: 1781.7350104
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7339598, upper bound: 1781.7338818
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7339598, upper bound: 1781.7350928
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7337907, upper bound: 1781.7332026
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7337907, upper bound: 1781.7337172
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7332198, upper bound: 1781.7351933
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7341542, upper bound: 1781.7351169
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7337907, upper bound: 1781.7332026
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7337907, upper bound: 1781.7337172
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7331658, upper bound: 1781.7345790
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7341542, upper bound: 1781.7348660
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7345950, upper bound: 1781.7343827
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7345950, upper bound: 1781.7345880
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7354919, upper bound: 1781.7354679
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7354919, upper bound: 1781.7355245
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7345950, upper bound: 1781.7343827
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7345950, upper bound: 1781.7345875
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7354919, upper bound: 1781.7354679
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -1781.7354919, upper bound: 1781.7355243

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -234.3851318, 951.7252808, -334.2373047, 1351.5371094, -1585.9221191, 1285.9626465
1: -377.8174438, 1052.7225342, -538.0744629, 1496.4943848, -1874.3117676, 1590.7967529
2: -281.3193054, 1213.5948486, -401.4847412, 1723.0072021, -2004.3264160, 1615.0795898
3: -607.2202759, 1082.6646729, -865.1986084, 1541.5124512, -2148.7326660, 1947.8632812
4: -484.4959717, 1135.6054688, -692.3477173, 1612.4166260, -2096.9125977, 1827.9531250

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7337171, upper bound: 1781.7326959
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7345096, upper bound: 1781.7325113
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7333204, upper bound: 1781.7311418
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 37

Time for candidate selection: 6.89 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7346956, upper bound: 1781.7333570
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7347392, upper bound: 1781.7333570
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -234.3851318, 951.7252808, -334.4379578, 1351.7272949, -1586.1123047, 1286.1632080
1: -377.8174438, 1052.7225342, -538.3348389, 1497.0654297, -1874.8826904, 1591.0573730
2: -281.3193054, 1213.5948486, -401.8078918, 1723.3419189, -2004.6612549, 1615.4027100
3: -607.2202759, 1082.6646729, -865.7811279, 1542.0057373, -2149.2258301, 1948.4458008
4: -484.4959717, 1135.6054688, -693.0050049, 1612.6823730, -2097.1782227, 1828.6104736

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7337171, upper bound: 1781.7329019
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7345096, upper bound: 1781.7336123
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7333204, upper bound: 1781.7316677
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 37

Time for candidate selection: 7.10 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7346956, upper bound: 1781.7342698
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7347392, upper bound: 1781.7342698
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -233.2238312, 946.9755859, -334.2373047, 1351.5371094, -1584.7608643, 1281.2128906
1: -375.9743347, 1047.7333984, -538.0744629, 1496.4943848, -1872.4687500, 1585.8076172
2: -280.0008850, 1207.5147705, -401.4847412, 1723.0072021, -2003.0080566, 1608.9995117
3: -604.5816040, 1076.7183838, -865.1986084, 1541.5124512, -2146.0939941, 1941.9169922
4: -482.0498657, 1129.5614014, -692.3477173, 1612.4166260, -2094.4665527, 1821.9090576

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7332586, upper bound: 1781.7322727
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7345151, upper bound: 1781.7325278
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7333676, upper bound: 1781.7311512
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 37

Time for candidate selection: 7.01 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7347285, upper bound: 1781.7333749
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7347460, upper bound: 1781.7333749
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -233.2238312, 946.9755859, -334.4379578, 1351.7272949, -1584.9511719, 1281.4134521
1: -375.9743347, 1047.7333984, -538.3348389, 1497.0654297, -1873.0397949, 1586.0682373
2: -280.0008850, 1207.5147705, -401.8078918, 1723.3419189, -2003.3427734, 1609.3226318
3: -604.5816040, 1076.7183838, -865.7811279, 1542.0057373, -2146.5871582, 1942.4995117
4: -482.0498657, 1129.5614014, -693.0050049, 1612.6823730, -2094.7316895, 1822.5664062

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7332586, upper bound: 1781.7322727
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7345151, upper bound: 1781.7332075
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7333676, upper bound: 1781.7313348
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 37

Time for candidate selection: 7.42 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7347285, upper bound: 1781.7339104
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7347460, upper bound: 1781.7336654
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -233.2722778, 947.1171265, -344.6470642, 1391.9061279, -1625.1782227, 1291.7639160
1: -375.8612366, 1047.7353516, -555.1515503, 1541.7329102, -1917.5941162, 1602.8867188
2: -279.8207703, 1207.8197021, -414.2575378, 1774.3829346, -2054.2036133, 1622.0772705
3: -604.0257568, 1077.4395752, -892.3187256, 1589.7650146, -2193.7905273, 1969.7583008
4: -482.2726440, 1129.9818115, -714.2735596, 1662.0943604, -2144.3666992, 1844.2550049

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7336100, upper bound: 1781.7317434
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7334302, upper bound: 1781.7316569
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 25

Time for candidate selection: 6.66 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7346956, upper bound: 1781.7339652
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7349813, upper bound: 1781.7339652
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -233.2722778, 947.1171265, -345.4939880, 1394.3183594, -1627.5905762, 1292.6110840
1: -375.8612366, 1047.7353516, -556.4404907, 1544.9345703, -1920.7957764, 1604.1757812
2: -279.8207703, 1207.8197021, -415.3844299, 1777.5404053, -2057.3610840, 1623.2041016
3: -604.0257568, 1077.4395752, -894.6403198, 1593.1328125, -2197.1582031, 1972.0798340
4: -482.2726440, 1129.9818115, -716.3289795, 1665.2487793, -2147.5209961, 1846.3106689

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7336100, upper bound: 1781.7317434
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7334302, upper bound: 1781.7317255
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 25

Time for candidate selection: 6.54 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7349390, upper bound: 1781.7342239
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7349813, upper bound: 1781.7342239
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -232.1515045, 942.6974487, -344.6470642, 1391.9061279, -1624.0574951, 1287.3442383
1: -374.1054993, 1043.0482178, -555.1515503, 1541.7329102, -1915.8383789, 1598.1994629
2: -278.5664673, 1202.1804199, -414.2575378, 1774.3829346, -2052.9494629, 1616.4379883
3: -601.5136108, 1071.8288574, -892.3187256, 1589.7650146, -2191.2785645, 1964.1475830
4: -479.9260864, 1124.3051758, -714.2735596, 1662.0943604, -2142.0202637, 1838.5784912

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7332586, upper bound: 1781.7314912
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7335073, upper bound: 1781.7316656
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 25

Time for candidate selection: 6.83 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7349547, upper bound: 1781.7339995
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7349703, upper bound: 1781.7339995
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -232.1515045, 942.6974487, -345.4939880, 1394.3183594, -1626.4698486, 1288.1914062
1: -374.1054993, 1043.0482178, -556.4404907, 1544.9345703, -1919.0400391, 1599.4886475
2: -278.5664673, 1202.1804199, -415.3844299, 1777.5404053, -2056.1066895, 1617.5648193
3: -601.5136108, 1071.8288574, -894.6403198, 1593.1328125, -2194.6464844, 1966.4692383
4: -479.9260864, 1124.3051758, -716.3289795, 1665.2487793, -2145.1745605, 1840.6340332

Time for backsubstitution: 1.50 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=2073.053466796875
rel_dist={0: [-1781.7388006996591, 1781.738800699659]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1134.79 seconds
