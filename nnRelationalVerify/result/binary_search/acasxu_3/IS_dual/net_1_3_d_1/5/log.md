## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_3.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 4551.82301297592


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953)
1: (-2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344)
2: (-3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344)
3: (-1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836)
4: (-3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328)

## BASE Result
execution time: IAR + LP analysis = 1.55 + 2.09 = 3.65 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -4552.2890012, upper bound: 4552.2890012


# Binary Search by BASE starts (time budget: 1196.35 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=5687.5751953125
rel_dist={0: [-4552.2878636421, 4552.287863642101]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=5687.5751953125
rel_dist={0: [-4552.283859528013, 4552.283859528014]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=5687.5751953125
rel_dist={0: [-4552.279487443704, 4552.279487443706]}

## Binary search (step 3) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=5687.5751953125
rel_dist={0: [-4552.274177015944, 4552.274177015945]}

## Binary search (step 4) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=5687.5751953125
rel_dist={0: [-4552.269705383769, 4552.269705383769]}

## Binary search (step 5) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=5687.5751953125
rel_dist={0: [-4552.266690988818, 4552.266690988818]}

## Binary search (step 6) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=5687.5751953125
rel_dist={0: [-4552.265148835347, 4552.265148835348]}

## Binary search (step 7) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=5687.5751953125
rel_dist={0: [-4552.264371856103, 4552.264371856105]}

## Binary search (step 8) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=5687.5751953125
rel_dist={0: [-4552.263982959417, 4552.263982959417]}

## Binary search (step 9) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=5687.5751953125
rel_dist={0: [-4552.263788511317, 4552.263788511316]}

## Binary search (step 10) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=5687.5751953125
rel_dist={0: [-4552.263681482938, 4552.263681482938]}

## Binary search (step 11) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=5687.5751953125
rel_dist={0: [-4552.26362479584, 4552.26362479584]}

## Binary search (step 12) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=5687.5751953125
rel_dist={0: [-4552.263594127799, 4552.263594127799]}

## Binary search (step 13) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=5687.5751953125
rel_dist={0: [-4552.263578672278, 4552.2635786722785]}

## Binary search (step 14) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=5687.5751953125
rel_dist={0: [-4552.263570946265, 4552.263570944802]}

## Binary search (step 15) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=5687.5751953125
rel_dist={0: [-4552.263567081615, 4552.2635670963]}

## Binary search (step 16) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=5687.5751953125
rel_dist={0: [-4552.263565151061, 4552.263565152603]}

## Binary search (step 17) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=5687.5751953125
rel_dist={0: [-4552.2635641832, 4552.263564185958]}

## Binary search (step 18) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=5687.5751953125
rel_dist={0: [-4552.263564553653, 4552.263563751329]}

## Binary search (step 19) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 19): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=5687.5751953125
rel_dist={0: [-4552.263576470345, 4552.26356620421]}

## Binary Search Result
Binary search time: 75.47 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1120.88 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2858669, upper bound: 4552.2771766
time: 1.56 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2756825, upper bound: 4552.2756825
time: 0.86 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.59 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.59
Output dim: 0, lower bound: -4552.2858669, upper bound: 4552.2771766
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.59
Output dim: 0, lower bound: -4552.2756825, upper bound: 4552.2756825

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -2987.9899902, 2430.9243164, -3136.6748047, 2550.9006348, -5538.8906250, 5567.5991211
1: -2393.8061523, 2341.6025391, -2513.2941895, 2457.4460449, -4851.2519531, 4854.8964844
2: -3424.2626953, 2546.4648438, -3595.4755859, 2672.1398926, -6096.4023438, 6141.9404297
3: -1341.7878418, 3393.8977051, -1407.5072021, 3564.2558594, -4906.0439453, 4801.4047852
4: -3810.2302246, 2503.8750000, -3999.4067383, 2627.0041504, -6437.2343750, 6503.2817383

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2110800, upper bound: 4552.1950579
time: 1.00 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905379, upper bound: 4552.1880421
time: 1.07 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -5188.6577148, 4191.8315430, -3111.8627930, 2531.4621582, -7720.1201172, 7303.6943359
1: -4159.4165039, 4035.3022461, -2493.2846680, 2438.7373047, -6598.1538086, 6528.5869141
2: -5950.7929688, 4386.3300781, -3566.7409668, 2651.7770996, -8602.5683594, 7953.0708008
3: -2313.8149414, 5892.1474609, -1396.7198486, 3536.2380371, -5850.0527344, 7288.8662109
4: -6623.0805664, 4320.7236328, -3967.7014160, 2606.9890137, -9230.0693359, 8288.4248047

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2755934, upper bound: 4552.2756825
time: 0.90 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2755934, upper bound: 4552.2756825
time: 0.93 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.44 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 3.44
Output dim: 0, lower bound: -4552.2110800, upper bound: 4552.1950579
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 3.44
Output dim: 0, lower bound: -4552.1905379, upper bound: 4552.1880421
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.44
Output dim: 0, lower bound: -4552.2755934, upper bound: 4552.2756825
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.44
Output dim: 0, lower bound: -4552.2755934, upper bound: 4552.2756825

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -3136.6748047, 2550.9006348, -5503.5830078, 5539.4765625
1: -2365.3845215, 2314.5366211, -2513.2941895, 2457.4460449, -4822.8305664, 4827.8310547
2: -3383.4587402, 2517.0407715, -3595.4755859, 2672.1398926, -6055.5976562, 6112.5151367
3: -1326.4201660, 3353.4309082, -1407.5072021, 3564.2558594, -4890.6752930, 4760.9379883
4: -3765.1281738, 2475.0305176, -3999.4067383, 2627.0041504, -6392.1323242, 6474.4375000

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2085840, upper bound: 4552.1950579
time: 0.85 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2085840, upper bound: 4552.1950579
time: 0.98 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -4203.4038086, 3378.4265137, -3126.3857422, 2542.5666504, -6745.9707031, 6504.8120117
1: -3380.6787109, 3251.6174316, -2505.0324707, 2449.3735352, -5830.0512695, 5756.6484375
2: -4848.3349609, 3525.0837402, -3583.6665039, 2663.3801270, -7511.7148438, 7108.7490234
3: -1855.3668213, 4787.9809570, -1402.9553223, 3552.5341797, -5407.9008789, 6190.9360352
4: -5372.3583984, 3470.2687988, -3986.2810059, 2618.4418945, -7990.8002930, 7456.5498047

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905380, upper bound: 4552.1856700
time: 0.76 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905380, upper bound: 4552.1880421
time: 0.64 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -5188.6577148, 4191.8315430, -2987.9899902, 2430.9243164, -7619.5820312, 7179.8212891
1: -4159.4165039, 4035.3022461, -2393.8061523, 2341.6025391, -6501.0190430, 6429.1083984
2: -5950.7929688, 4386.3300781, -3424.2626953, 2546.4648438, -8497.2558594, 7810.5922852
3: -2313.8149414, 5892.1474609, -1341.7878418, 3393.8977051, -5707.7128906, 7233.9355469
4: -6623.0805664, 4320.7236328, -3810.2302246, 2503.8750000, -9126.9550781, 8130.9531250

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1942569
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1905380
time: 0.85 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -5188.6577148, 4191.8315430, -5188.6577148, 4191.8315430, -9380.4892578, 9380.4892578
1: -4159.4165039, 4035.3022461, -4159.4165039, 4035.3022461, -8194.7187500, 8194.7187500
2: -5950.7929688, 4386.3300781, -5950.7929688, 4386.3300781, -10334.5771484, 10334.5771484
3: -2313.8149414, 5892.1474609, -2313.8149414, 5892.1474609, -8205.9628906, 8205.9628906
4: -6623.0805664, 4320.7236328, -6623.0805664, 4320.7236328, -10938.7910156, 10938.7919922

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1942569
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1905380
time: 0.88 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.48 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 0, lower bound: -4552.2085840, upper bound: 4552.1950579
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 0, lower bound: -4552.2085840, upper bound: 4552.1950579
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 0, lower bound: -4552.1905380, upper bound: 4552.1856700
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 0, lower bound: -4552.1905380, upper bound: 4552.1880421
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1942569
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1905380
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1942569
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1905380

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -2987.9899902, 2430.9243164, -5383.6064453, 5390.7915039
1: -2365.3845215, 2314.5366211, -2393.8061523, 2341.6025391, -4706.9873047, 4708.3427734
2: -3383.4587402, 2517.0407715, -3424.2626953, 2546.4648438, -5929.9228516, 5941.3027344
3: -1326.4201660, 3353.4309082, -1341.7878418, 3393.8977051, -4720.3173828, 4695.2187500
4: -3765.1281738, 2475.0305176, -3810.2302246, 2503.8750000, -6269.0029297, 6285.2602539

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2034861, upper bound: 4551.4744295
time: 1.27 seconds

## Relational analysis of IS_A1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2003084, upper bound: 4552.1943617
time: 0.79 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -5188.6577148, 4191.8315430, -7144.5136719, 7591.4589844
1: -2365.3845215, 2314.5366211, -4159.4165039, 4035.3022461, -6400.6865234, 6473.9531250
2: -3383.4587402, 2517.0407715, -5950.7929688, 4386.3300781, -7769.7875977, 8467.8320312
3: -1326.4201660, 3353.4309082, -2313.8149414, 5892.1474609, -7218.5668945, 5667.2460938
4: -3765.1281738, 2475.0305176, -6623.0805664, 4320.7236328, -8085.8510742, 9098.1093750

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2034861, upper bound: 4551.4744295
time: 0.92 seconds

## Relational analysis of IS_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2003084, upper bound: 4552.1943617
time: 0.84 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -4186.0019531, 3365.2031250, -3099.8066406, 2521.3447266, -6707.3461914, 6465.0097656
1: -3366.3283691, 3238.8005371, -2483.6220703, 2428.9890137, -5795.3173828, 5722.4208984
2: -4827.5234375, 3511.4433594, -3552.8640137, 2641.1804199, -7468.7031250, 7064.3076172
3: -1848.4030762, 4767.6972656, -1391.2972412, 3521.8344727, -5370.2368164, 6158.9931641
4: -5349.6049805, 3457.0251465, -3952.2563477, 2596.6862793, -7946.2910156, 7409.2812500

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1856700
time: 0.78 seconds

## Relational analysis of IS_A1_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1856700
time: 0.95 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -4267.7421875, 3427.3356934, -4396.1411133, 3531.7072754, -7799.4492188, 7823.4760742
1: -3433.8293457, 3298.9086914, -3536.9965820, 3399.6027832, -6833.4311523, 6835.9047852
2: -4925.3828125, 3575.4016113, -5073.0039062, 3685.0007324, -8610.3818359, 8648.4052734
3: -1881.0106201, 4863.0942383, -1939.8779297, 5009.5537109, -6890.5644531, 6802.9721680
4: -5456.5566406, 3519.1264648, -5620.0600586, 3627.3405762, -9083.8974609, 9139.1845703

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1876281
time: 0.79 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1876281
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -2987.9899902, 2430.9243164, -7589.3227539, 7155.5288086
1: -4135.0488281, 4011.9240723, -2393.8061523, 2341.6025391, -6476.6513672, 6405.7290039
2: -5915.9174805, 4360.9184570, -3424.2626953, 2546.4648438, -8462.3818359, 7785.1806641
3: -2300.4028320, 5857.5903320, -1341.7878418, 3393.8977051, -5694.3007812, 7199.3779297
4: -6584.4262695, 4295.7910156, -3810.2302246, 2503.8750000, -9088.3007812, 8106.0195312

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1938141
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1892375
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6427.7465820, 5168.6801758, -2976.7226562, 2421.7700195, -8849.5146484, 8145.4013672
1: -5157.6206055, 4970.1308594, -2384.7473145, 2332.7238770, -7490.3447266, 7354.8779297
2: -7386.5043945, 5401.5595703, -3411.3022461, 2536.8029785, -9923.3076172, 8812.8593750
3: -2849.8481445, 7301.7617188, -1336.7360840, 3381.0124512, -6230.8603516, 8638.4980469
4: -8201.5048828, 5324.3559570, -3795.8500977, 2494.4553223, -10695.9589844, 9120.2060547

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1884626
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1905380
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -5188.6577148, 4191.8315430, -9350.2294922, 9356.1972656
1: -4135.0488281, 4011.9240723, -4159.4165039, 4035.3022461, -8170.3510742, 8171.3408203
2: -5915.9174805, 4360.9184570, -5950.7929688, 4386.3300781, -10299.6093750, 10309.1445312
3: -2300.4028320, 5857.5903320, -2313.8149414, 5892.1474609, -8192.5507812, 8171.4052734
4: -6584.4262695, 4295.7910156, -6623.0805664, 4320.7236328, -10900.1123047, 10913.8359375

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1794140, upper bound: 4551.4731700
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1762363, upper bound: 4552.1932545
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6425.8520508, 5167.2119141, -5175.4057617, 4181.1166992, -10606.9687500, 10338.9990234
1: -5156.0722656, 4968.7270508, -4148.7426758, 4024.9902344, -9181.0615234, 9116.1416016
2: -7384.2504883, 5400.0463867, -5935.5375977, 4375.1381836, -11751.1933594, 11326.3291016
3: -2849.0493164, 7299.5883789, -2307.9543457, 5877.0166016, -8722.6455078, 9587.4726562
4: -8199.0634766, 5322.8666992, -6606.1806641, 4309.7270508, -12503.6699219, 11916.4023438

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1881660
time: 1.26 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1905380
time: 1.06 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.15 seconds
IS_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -4552.2034861, upper bound: 4551.4744295
IS_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -4552.2003084, upper bound: 4552.1943617
IS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -4552.2034861, upper bound: 4551.4744295
IS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -4552.2003084, upper bound: 4552.1943617
IS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1856700
IS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1856700
IS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1876281
IS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1876281
IS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1938141
IS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1892375
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1884626
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1905380
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -4552.1794140, upper bound: 4551.4731700
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -4552.1762363, upper bound: 4552.1932545
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1881660
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1905380

## BFS IS instance: IS_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -2864.9853516, 2335.5559082, -5288.2382812, 5267.7871094
1: -2365.3845215, 2314.5366211, -2295.3029785, 2250.7009277, -4616.0854492, 4609.8393555
2: -3383.4587402, 2517.0407715, -3284.9338379, 2445.8928223, -5829.3505859, 5801.9736328
3: -1326.4201660, 3353.4309082, -1288.4378662, 3259.7199707, -4586.1396484, 4641.8686523
4: -3765.1281738, 2475.0305176, -3655.1245117, 2403.8120117, -6168.9389648, 6130.1552734

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B1_B1_A1

### Relational analysis result of IS_A1_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4375556, upper bound: 4551.4744295
time: 0.83 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2

### Relational analysis result of IS_A1_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4375556, upper bound: 4551.4744295
time: 0.73 seconds

## BFS IS instance: IS_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -2890.1743164, 2353.1525879, -5305.8344727, 5292.9755859
1: -2365.3845215, 2314.5366211, -2314.7424316, 2266.5371094, -4631.9218750, 4629.2788086
2: -3383.4587402, 2517.0407715, -3311.0268555, 2464.3937988, -5847.8510742, 5828.0668945
3: -1326.4201660, 3353.4309082, -1298.1706543, 3283.7622070, -4610.1811523, 4651.6005859
4: -3765.1281738, 2475.0305176, -3685.6552734, 2423.2575684, -6188.3857422, 6160.6855469

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_A1_B1_B2_B1

### Relational analysis result of IS_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1998742, upper bound: 4552.1943617
time: 0.84 seconds

## Relational analysis of IS_A1_A1_B1_B2_B2

### Relational analysis result of IS_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2003084, upper bound: 4552.1941248
time: 0.85 seconds

## BFS IS instance: IS_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -5059.5898438, 4091.7714844, -7044.4536133, 7462.3916016
1: -2365.3845215, 2314.5366211, -4055.8063965, 3939.3488770, -6304.7333984, 6370.3427734
2: -3383.4587402, 2517.0407715, -5803.9467773, 4281.0913086, -7664.5488281, 8320.9863281
3: -1326.4201660, 3353.4309082, -2258.0759277, 5750.6269531, -7077.0468750, 5611.5063477
4: -3765.1281738, 2475.0305176, -6459.7363281, 4216.7597656, -7981.8876953, 8934.7666016

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2051240, upper bound: 4551.0306479
time: 0.86 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2

### Relational analysis result of IS_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2043651, upper bound: 4551.4744237
time: 0.73 seconds

## BFS IS instance: IS_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -5104.3461914, 4125.3242188, -7078.0063477, 7507.1474609
1: -2365.3845215, 2314.5366211, -4091.1743164, 3971.0102539, -6336.3945312, 6405.7109375
2: -3383.4587402, 2517.0407715, -5852.9707031, 4316.5053711, -7699.9633789, 8370.0087891
3: -1326.4201660, 3353.4309082, -2276.7976074, 5797.0576172, -7123.4775391, 5630.2285156
4: -3765.1281738, 2475.0305176, -6515.3515625, 4252.2592773, -8017.3862305, 8990.3808594

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2030297, upper bound: 4551.9559997
time: 0.73 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2018617, upper bound: 4552.1943493
time: 0.80 seconds

## BFS IS instance: IS_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -4185.1938477, 3364.5900879, -2952.6821289, 2402.8015137, -6587.9951172, 6317.2719727
1: -3365.6628418, 3238.2065430, -2365.3845215, 2314.5366211, -5680.1992188, 5603.5908203
2: -4826.5581055, 3510.8107910, -3383.4587402, 2517.0407715, -7343.5971680, 6894.2695312
3: -1848.0799561, 4766.7568359, -1326.4201660, 3353.4309082, -5201.5102539, 6093.1767578
4: -5348.5498047, 3456.4106445, -3765.1281738, 2475.0305176, -7823.5800781, 7221.5380859

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B1_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1938350, upper bound: 4551.3963185
time: 0.85 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2

### Relational analysis result of IS_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1930896, upper bound: 4552.1762363
time: 0.76 seconds

## BFS IS instance: IS_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -4182.2895508, 3362.3830566, -5158.3984375, 4167.5395508, -8349.8291016, 8520.7812500
1: -3363.2678223, 3236.0671387, -4135.0488281, 4011.9240723, -7375.1909180, 7371.1162109
2: -4823.0864258, 3508.5339355, -5915.9174805, 4360.9184570, -9184.0048828, 9424.4511719
3: -1846.9169922, 4763.3745117, -2300.4028320, 5857.5903320, -7704.5073242, 7063.7773438
4: -5344.7543945, 3454.1999512, -6584.4262695, 4295.7910156, -9640.5439453, 10038.6259766

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B1_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1938350, upper bound: 4551.3963185
time: 0.83 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2

### Relational analysis result of IS_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1930896, upper bound: 4552.1762363
time: 0.81 seconds

## BFS IS instance: IS_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -4267.7421875, 3427.3356934, -4267.7421875, 3427.3356934, -7695.0771484, 7695.0771484
1: -3433.8293457, 3298.9086914, -3433.8293457, 3298.9086914, -6732.7377930, 6732.7377930
2: -4925.3828125, 3575.4016113, -4925.3828125, 3575.4016113, -8500.7832031, 8500.7841797
3: -1881.0106201, 4863.0942383, -1881.0106201, 4863.0942383, -6744.1049805, 6744.1049805
4: -5456.5566406, 3519.1264648, -5456.5566406, 3519.1264648, -8975.6835938, 8975.6835938

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B2_B1_A1

### Relational analysis result of IS_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1871792
time: 0.86 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2

### Relational analysis result of IS_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844829, upper bound: 4552.1837217
time: 0.75 seconds

## BFS IS instance: IS_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -4267.7421875, 3427.3356934, -6465.6679688, 5197.9145508, -9465.6562500, 9893.0039062
1: -3433.8293457, 3298.9086914, -5188.6606445, 4998.1103516, -8431.9394531, 8487.5683594
2: -4925.3828125, 3575.4016113, -7431.6689453, 5431.6601562, -10357.0410156, 11007.0703125
3: -1881.0106201, 4863.0942383, -2865.7429199, 7345.2402344, -9226.2509766, 7724.6606445
4: -5456.5566406, 3519.1264648, -8250.3779297, 5354.0131836, -10810.5673828, 11769.5009766

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1876606, upper bound: 4551.4630992
time: 0.94 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844829, upper bound: 4552.1837216
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -2987.9899902, 2430.9243164, -7028.0419922, 6695.5332031
1: -3687.0437012, 3569.1484375, -2393.8061523, 2341.6025391, -6028.6455078, 5962.9545898
2: -5276.2031250, 3879.7956543, -3424.2626953, 2546.4648438, -7822.6679688, 7304.0576172
3: -2047.8087158, 5221.5805664, -1341.7878418, 3393.8977051, -5441.7060547, 6563.3681641
4: -5870.9716797, 3822.0715332, -3810.2302246, 2503.8750000, -8374.8466797, 7632.3007812

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.1882687, upper bound: 4552.1934081
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A1_A1_A2

### Relational analysis result of IS_A2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1513498, upper bound: 4552.1927694
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -2987.9899902, 2430.9243164, -7534.8857422, 7111.2729492
1: -4091.2307129, 3969.2382812, -2393.8061523, 2341.6025391, -6432.8330078, 6363.0439453
2: -5853.0830078, 4314.3935547, -3424.2626953, 2546.4648438, -8399.5478516, 7738.6557617
3: -2275.4304199, 5795.1562500, -1341.7878418, 3393.8977051, -5669.3276367, 7136.9443359
4: -6514.6850586, 4250.0151367, -3810.2302246, 2503.8750000, -9018.5605469, 8060.2446289

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_A2_A1

### Relational analysis result of IS_A2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.3963185, upper bound: 4552.1888561
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_A2_A2

### Relational analysis result of IS_A2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1762363, upper bound: 4552.1865734
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6414.4453125, 5158.3549805, -2952.6821289, 2402.8015137, -8817.2470703, 8111.0371094
1: -5146.8022461, 4960.2563477, -2365.3845215, 2314.5366211, -7461.3388672, 7325.6406250
2: -7370.6704102, 5390.9252930, -3383.4587402, 2517.0407715, -9887.7109375, 8774.3837891
3: -2844.2338867, 7286.4809570, -1326.4201660, 3353.4309082, -6197.6650391, 8612.9013672
4: -8184.3613281, 5313.8818359, -3765.1281738, 2475.0305176, -10659.3916016, 9079.0097656

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9503000, upper bound: 4552.1876046
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1868456
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6465.6679688, 5197.9145508, -4267.7421875, 3427.3356934, -9893.0039062, 9465.6562500
1: -5188.6606445, 4998.1103516, -3433.8293457, 3298.9086914, -8487.5683594, 8431.9394531
2: -7431.6689453, 5431.6601562, -4925.3828125, 3575.4016113, -11007.0703125, 10357.0410156
3: -2865.7429199, 7345.2402344, -1881.0106201, 4863.0942383, -7724.6611328, 9226.2509766
4: -8250.3779297, 5354.0131836, -5456.5566406, 3519.1264648, -11769.5009766, 10810.5673828

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9503000, upper bound: 4552.1892924
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1886312
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -5059.5898438, 4091.7714844, -9250.1699219, 9227.1289062
1: -4135.0488281, 4011.9240723, -4055.8063965, 3939.3488770, -8074.3974609, 8067.7304688
2: -5915.9174805, 4360.9184570, -5803.9467773, 4281.0913086, -10192.9169922, 10159.8750000
3: -2300.4028320, 5857.5903320, -2258.0759277, 5750.6269531, -8051.0297852, 8113.7324219
4: -6584.4262695, 4295.7910156, -6459.7363281, 4216.7597656, -10794.9550781, 10748.1210938

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1545275, upper bound: 4551.4726643
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1794140, upper bound: 4551.4667228
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -5104.3461914, 4125.3242188, -9283.7226562, 9271.8857422
1: -4135.0488281, 4011.9240723, -4091.1743164, 3971.0102539, -8106.0585938, 8103.0986328
2: -5915.9174805, 4360.9184570, -5852.9707031, 4316.5053711, -10227.7890625, 10212.7812500
3: -2300.4028320, 5857.5903320, -2276.7976074, 5797.0576172, -8097.4604492, 8132.1708984
4: -6584.4262695, 4295.7910156, -6515.3515625, 4252.2592773, -10829.7646484, 10807.4472656

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1513498, upper bound: 4552.1927694
time: 1.51 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1762363, upper bound: 4552.1865734
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6412.2436523, 5156.6450195, -5158.3984375, 4167.5395508, -10579.7832031, 10311.4404297
1: -5145.0195312, 4958.6206055, -4135.0488281, 4011.9240723, -9156.9433594, 9092.3554688
2: -7368.0488281, 5389.1655273, -5915.9174805, 4360.9184570, -11721.0742188, 11295.8056641
3: -2843.3049316, 7283.9501953, -2300.4028320, 5857.5903320, -8697.4072266, 9564.5478516
4: -8181.5249023, 5312.1459961, -6584.4262695, 4295.7910156, -12472.2968750, 11883.9404297

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1876606, upper bound: 4551.3996801
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1795978
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6465.6679688, 5197.9145508, -6465.6679688, 5197.9145508, -11659.2441406, 11659.2441406
1: -5188.6606445, 4998.1103516, -5188.6606445, 4998.1103516, -10185.4218750, 10185.4208984
2: -7431.6689453, 5431.6601562, -7431.6689453, 5431.6601562, -12847.5341797, 12847.5341797
3: -2865.7429199, 7345.2402344, -2865.7429199, 7345.2402344, -10186.5107422, 10186.5107422
4: -8250.3779297, 5354.0131836, -8250.3779297, 5354.0131836, -13591.3330078, 13591.3330078

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1898002
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1872592
time: 0.88 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.30 seconds
IS_A1_A1_B1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 0, lower bound: -4551.4375556, upper bound: 4551.4744295
IS_A1_A1_B1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 0, lower bound: -4551.4375556, upper bound: 4551.4744295
IS_A1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -4552.1998742, upper bound: 4552.1943617
IS_A1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -4552.2003084, upper bound: 4552.1941248
IS_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -4552.2051240, upper bound: 4551.0306479
IS_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -4552.2043651, upper bound: 4551.4744237
IS_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -4552.2030297, upper bound: 4551.9559997
IS_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -4552.2018617, upper bound: 4552.1943493
IS_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -4552.1938350, upper bound: 4551.3963185
IS_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -4552.1930896, upper bound: 4552.1762363
IS_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -4552.1938350, upper bound: 4551.3963185
IS_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -4552.1930896, upper bound: 4552.1762363
IS_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1871792
IS_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -4552.1844829, upper bound: 4552.1837217
IS_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -4552.1876606, upper bound: 4551.4630992
IS_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -4552.1844829, upper bound: 4552.1837216
IS_A2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -4551.1882687, upper bound: 4552.1934081
IS_A2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -4552.1513498, upper bound: 4552.1927694
IS_A2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -4551.3963185, upper bound: 4552.1888561
IS_A2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -4552.1762363, upper bound: 4552.1865734
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -4551.9503000, upper bound: 4552.1876046
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1868456
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -4551.9503000, upper bound: 4552.1892924
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1886312
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -4552.1545275, upper bound: 4551.4726643
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -4552.1794140, upper bound: 4551.4667228
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -4552.1513498, upper bound: 4552.1927694
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -4552.1762363, upper bound: 4552.1865734
IS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -4552.1876606, upper bound: 4551.3996801
IS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1795978
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1898002
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1872592

## BFS IS instance: IS_A1_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -2946.0878906, 2397.5485840, -2890.3549805, 2354.6845703, -5300.7719727, 5287.9013672
1: -2360.0886230, 2309.4670410, -2314.6940918, 2268.1701660, -4628.2587891, 4624.1601562
2: -3375.8811035, 2511.5446777, -3310.4218750, 2465.3471680, -5841.2275391, 5821.9667969
3: -1323.5562744, 3345.9208984, -1298.0891113, 3286.2832031, -4609.8393555, 4644.0092773
4: -3756.7243652, 2469.6533203, -3685.1674805, 2425.0095215, -6181.7333984, 6154.8193359

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B1_B2_B1_A1

### Relational analysis result of IS_A1_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1691642, upper bound: 4552.1943617
time: 0.87 seconds

## Relational analysis of IS_A1_A1_B1_B2_B1_A2

### Relational analysis result of IS_A1_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1998742, upper bound: 4552.1943493
time: 0.90 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -2830.7468262, 2305.1364746, -5257.8183594, 5233.5483398
1: -2365.3845215, 2314.5366211, -2267.0932617, 2220.3234863, -4585.7080078, 4581.6298828
2: -3383.4587402, 2517.0407715, -3242.8298340, 2413.6757812, -5797.1333008, 5759.8701172
3: -1326.4201660, 3353.4309082, -1271.3063965, 3216.4094238, -4542.8295898, 4624.7373047
4: -3765.1281738, 2475.0305176, -3609.7048340, 2373.3562012, -6138.4829102, 6084.7353516

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B1_B2_B2_A1

### Relational analysis result of IS_A1_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4369877, upper bound: 4552.1941249
time: 0.90 seconds

## Relational analysis of IS_A1_A1_B1_B2_B2_A2

### Relational analysis result of IS_A1_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4369877, upper bound: 4552.1917008
time: 0.82 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -4511.3120117, 3641.1752930, -6593.8574219, 6914.1132812
1: -2365.3845215, 2314.5366211, -3618.0803223, 3505.7343750, -5871.1191406, 5932.6171875
2: -3383.4587402, 2517.0407715, -5178.7822266, 3809.7700195, -7193.2275391, 7695.8222656
3: -1326.4201660, 3353.4309082, -2010.6011963, 5128.5458984, -6454.9653320, 5364.0307617
4: -3765.1281738, 2475.0305176, -5762.4921875, 3752.6533203, -7517.7802734, 8237.5224609

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B1_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2051240, upper bound: 4551.0306479
time: 0.73 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_B2

### Relational analysis result of IS_A1_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2051240, upper bound: 4551.0306479
time: 0.76 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -5005.4360352, 4047.6650391, -7000.3471680, 7408.2373047
1: -2365.3845215, 2314.5366211, -4012.2070312, 3896.7814941, -6262.1660156, 6326.7436523
2: -3383.4587402, 2517.0407715, -5741.3471680, 4234.6699219, -7618.1279297, 8258.3857422
3: -1326.4201660, 3353.4309082, -2233.0949707, 5688.2758789, -7014.6958008, 5586.5258789
4: -3765.1281738, 2475.0305176, -6390.2583008, 4171.0961914, -7936.2241211, 8865.2871094

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2043651, upper bound: 4551.4062316
time: 0.83 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2043651, upper bound: 4551.4744237
time: 0.75 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -4542.9423828, 3664.9570312, -6617.6391602, 6945.7441406
1: -2365.3845215, 2314.5366211, -3643.1877441, 3527.8876953, -5893.2724609, 5957.7246094
2: -3383.4587402, 2517.0407715, -5213.2275391, 3835.0407715, -7218.4990234, 7730.2670898
3: -1326.4201660, 3353.4309082, -2024.0731201, 5160.9624023, -6487.3818359, 5377.5039062
4: -3765.1281738, 2475.0305176, -5801.8403320, 3778.2233887, -7543.3500977, 8276.8710938

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2030297, upper bound: 4551.9559997
time: 0.92 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_B2

### Relational analysis result of IS_A1_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2030297, upper bound: 4551.9559997
time: 0.81 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -5049.5405273, 4080.6557617, -7033.3378906, 7452.3417969
1: -2365.3845215, 2314.5366211, -4047.0185547, 3927.9313965, -6293.3154297, 6361.5551758
2: -3383.4587402, 2517.0407715, -5789.6328125, 4269.5668945, -7653.0244141, 8306.6728516
3: -1326.4201660, 3353.4309082, -2251.6179199, 5734.0175781, -7060.4370117, 5605.0488281
4: -3765.1281738, 2475.0305176, -6445.0820312, 4206.0708008, -7971.1982422, 8920.1123047

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0018232, upper bound: 4552.1910870
time: 0.81 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2002294, upper bound: 4552.1928830
time: 0.84 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -4182.5283203, 3362.5644531, -2829.3090820, 2307.1660156, -6489.6943359, 6191.8735352
1: -3363.4648438, 3236.2429199, -2266.5866699, 2223.4016113, -5586.8657227, 5502.8295898
2: -4823.3715820, 3508.7216797, -3243.7138672, 2416.2275391, -7239.5966797, 6752.4355469
3: -1847.0126953, 4763.6523438, -1272.9768066, 3218.7609863, -5065.7729492, 6036.6284180
4: -5345.0659180, 3454.3820801, -3609.5827637, 2374.6940918, -7719.7597656, 7063.9633789

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B1_B1_B1_A1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4744292, upper bound: 4551.4375556
time: 0.80 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A2

### Relational analysis result of IS_A1_A2_B1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4744292, upper bound: 4551.4375556
time: 0.76 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -4183.9614258, 3363.6533203, -2854.9377441, 2325.1379395, -6509.0991211, 6218.5908203
1: -3364.6457520, 3237.2985840, -2286.3728027, 2239.5864258, -5604.2324219, 5523.6713867
2: -4825.0844727, 3509.8447266, -3270.2888184, 2435.0783691, -7260.1625977, 6780.1333008
3: -1847.5864258, 4765.3212891, -1282.8604736, 3243.4042969, -5090.9907227, 6048.1816406
4: -5346.9399414, 3455.4724121, -3640.6374512, 2394.5166016, -7741.4550781, 7096.1098633

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B1_B1_B2_A1

### Relational analysis result of IS_A1_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4744292, upper bound: 4552.2003084
time: 0.77 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2_A2

### Relational analysis result of IS_A1_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4744292, upper bound: 4552.2003084
time: 0.94 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -4179.3085938, 3360.1145020, -5030.1420898, 4068.0788574, -8247.3876953, 8390.2568359
1: -3360.8107910, 3233.8684082, -4032.0764160, 3916.5642090, -7277.3750000, 7265.9448242
2: -4819.5239258, 3506.1950684, -5769.9741211, 4256.2983398, -9075.8222656, 9276.1679688
3: -1845.7227783, 4759.8994141, -2244.9921875, 5716.9531250, -7562.6757812, 7004.8393555
4: -5340.8598633, 3451.9284668, -6422.0834961, 4192.4262695, -9533.2851562, 9874.0117188

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B1_B2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1934080, upper bound: 4551.1882687
time: 0.76 seconds

## Relational analysis of IS_A1_A2_B1_B2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1888560, upper bound: 4551.3963185
time: 0.96 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -4181.0312500, 3361.4260254, -5074.3295898, 4101.2333984, -8282.2636719, 8435.7558594
1: -3362.2307129, 3235.1391602, -4067.0036621, 3947.8349609, -7310.0649414, 7302.1425781
2: -4821.5830078, 3507.5468750, -5818.3769531, 4291.3007812, -9112.8837891, 9325.9238281
3: -1846.4133301, 4761.9082031, -2263.4909668, 5762.7929688, -7609.2060547, 7025.0102539
4: -5343.1103516, 3453.2414551, -6477.0117188, 4227.5278320, -9570.6386719, 9930.2519531

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B1_B2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1927693, upper bound: 4552.1513498
time: 0.78 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2_B2

### Relational analysis result of IS_A1_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1865733, upper bound: 4552.1762363
time: 0.99 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -4175.5634766, 3350.7539062, -4267.7421875, 3427.3356934, -7602.8984375, 7618.4960938
1: -3362.1945801, 3227.0385742, -3433.8293457, 3298.9086914, -6661.1030273, 6660.8671875
2: -4825.2534180, 3494.4333496, -4925.3828125, 3575.4016113, -8400.6552734, 8419.8144531
3: -1836.1400146, 4765.6860352, -1881.0106201, 4863.0942383, -6699.2343750, 6646.6958008
4: -5343.5141602, 3436.8771973, -5456.5566406, 3519.1264648, -8862.6386719, 8893.4335938

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B2_B1_A1_B1

### Relational analysis result of IS_A1_A2_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4646324, upper bound: 4551.4630992
time: 1.01 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_B2

### Relational analysis result of IS_A1_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1837414
time: 0.77 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -4186.9238281, 3361.6271973, -4267.7421875, 3427.3356934, -7614.2583008, 7629.3691406
1: -3369.0449219, 3235.5971680, -3433.8293457, 3298.9086914, -6667.9536133, 6669.4262695
2: -4832.7299805, 3506.0024414, -4925.3828125, 3575.4016113, -8408.1318359, 8431.3847656
3: -1843.7967529, 4772.0375977, -1881.0106201, 4863.0942383, -6706.8911133, 6653.0483398
4: -5354.3334961, 3450.2895508, -5456.5566406, 3519.1264648, -8873.4599609, 8906.8457031

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B2_B1_A2_B1

### Relational analysis result of IS_A1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844829, upper bound: 4551.4630992
time: 1.21 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_B2

### Relational analysis result of IS_A1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1827801, upper bound: 4552.1837414
time: 0.79 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -4267.7421875, 3427.3356934, -6366.1660156, 5119.2890625, -9387.0302734, 9793.5019531
1: -3433.8293457, 3298.9086914, -5110.7729492, 4923.2451172, -8357.0742188, 8409.6816406
2: -4925.3828125, 3575.4016113, -7322.4482422, 5348.0849609, -10273.4677734, 10897.8486328
3: -1881.0106201, 4863.0942383, -2820.9536133, 7239.8808594, -9120.8203125, 7676.4799805
4: -5456.5566406, 3519.1264648, -8126.7597656, 5270.9204102, -10727.4765625, 11645.8867188

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B2_B2_B1_B1

### Relational analysis result of IS_A1_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1892986, upper bound: 4551.0207390
time: 0.87 seconds

## Relational analysis of IS_A1_A2_B2_B2_B1_B2

### Relational analysis result of IS_A1_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1885396, upper bound: 4551.4630992
time: 0.97 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -4267.7421875, 3427.3356934, -6390.3413086, 5137.2963867, -9405.0390625, 9817.6767578
1: -3433.8293457, 3298.9086914, -5128.1879883, 4939.5961914, -8373.4257812, 8427.0966797
2: -4925.3828125, 3575.4016113, -7344.9951172, 5367.6391602, -10293.0214844, 10920.3964844
3: -1881.0106201, 4863.0942383, -2831.8232422, 7260.3583984, -9141.3691406, 7687.5224609
4: -5456.5566406, 3519.1264648, -8154.4711914, 5291.2568359, -10747.8134766, 11673.5966797

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B2_B2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1872042, upper bound: 4551.9460957
time: 0.85 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1843335, upper bound: 4552.1837216
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -4483.1152344, 3618.3911133, -2987.9899902, 2430.9243164, -6914.0395508, 6606.3808594
1: -3595.3923340, 3483.8129883, -2393.8061523, 2341.6025391, -5936.9946289, 5877.6186523
2: -5146.2998047, 3785.9094238, -3424.2626953, 2546.4648438, -7692.7641602, 7210.1718750
3: -1998.0097656, 5096.2939453, -1341.7878418, 3393.8977051, -5391.9072266, 6438.0820312
4: -5726.4765625, 3729.2534180, -3810.2302246, 2503.8750000, -8230.3515625, 7539.4829102

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_A1_A1_B1

### Relational analysis result of IS_A2_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.1864078, upper bound: 4552.1893553
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A1_A1_A1_B2

### Relational analysis result of IS_A2_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.1870258, upper bound: 4552.1934081
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -4514.2919922, 3641.8427734, -2987.9899902, 2430.9243164, -6945.2163086, 6629.8320312
1: -3620.1120605, 3505.6416016, -2393.8061523, 2341.6025391, -5961.7148438, 5899.4477539
2: -5180.1918945, 3810.8364258, -3424.2626953, 2546.4648438, -7726.6562500, 7235.0991211
3: -2011.2974854, 5128.1811523, -1341.7878418, 3393.8977051, -5405.1953125, 6469.9687500
4: -5765.2221680, 3754.4975586, -3810.2302246, 2503.8750000, -8269.0957031, 7564.7270508

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_A1_A2_B1

### Relational analysis result of IS_A2_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1494888, upper bound: 4552.1885298
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_A1_A2_B2

### Relational analysis result of IS_A2_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1501069, upper bound: 4552.1925717
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -4976.5166016, 4024.3906250, -2987.9899902, 2430.9243164, -7407.4409180, 7012.3808594
1: -3988.8959961, 3874.3952637, -2393.8061523, 2341.6025391, -6330.4985352, 6268.2006836
2: -5707.9775391, 4210.3144531, -3424.2626953, 2546.4648438, -8254.4423828, 7634.5771484
3: -2220.2397461, 5655.1889648, -1341.7878418, 3393.8977051, -5614.1376953, 6996.9765625
4: -6353.2714844, 4147.1865234, -3810.2302246, 2503.8750000, -8857.1464844, 7957.4155273

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_A2_A1_B1

### Relational analysis result of IS_A2_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.3944576, upper bound: 4552.1843529
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_A2_A1_B2

### Relational analysis result of IS_A2_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.3950756, upper bound: 4552.1884367
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -5020.0449219, 4056.9860840, -2987.9899902, 2430.9243164, -7450.9692383, 7044.9755859
1: -4023.2673340, 3905.1572266, -2393.8061523, 2341.6025391, -6364.8701172, 6298.9633789
2: -5755.6445312, 4244.7993164, -3424.2626953, 2546.4648438, -8302.1074219, 7669.0620117
3: -2238.5415039, 5700.3437500, -1341.7878418, 3393.8977051, -5632.4394531, 7042.1318359
4: -6407.4130859, 4181.7666016, -3810.2302246, 2503.8750000, -8911.2880859, 7991.9965820

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_A2_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1743754, upper bound: 4552.1817034
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A1_A2_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1749934, upper bound: 4552.1857872
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5922.2187500, 4743.0864258, -2952.6821289, 2402.8015137, -8325.0205078, 7695.7685547
1: -4753.3393555, 4561.6894531, -2365.3845215, 2314.5366211, -7067.8759766, 6927.0742188
2: -6808.0859375, 4955.4916992, -3383.4587402, 2517.0407715, -9325.1250000, 8338.9501953
3: -2614.2287598, 6720.6738281, -1326.4201660, 3353.4309082, -5967.6591797, 8047.0937500
4: -7557.2294922, 4885.1381836, -3765.1281738, 2475.0305176, -10032.2568359, 8650.2666016

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.0306479, upper bound: 4552.2051240
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9559997, upper bound: 4552.2030297
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6310.1459961, 5077.5209961, -2952.6821289, 2402.8015137, -8712.9462891, 8030.2031250
1: -5062.7065430, 4882.0249023, -2365.3845215, 2314.5366211, -7377.2431641, 7247.4091797
2: -7249.5307617, 5306.7485352, -3383.4587402, 2517.0407715, -9766.5712891, 8690.2060547
3: -2799.9042969, 7168.0913086, -1326.4201660, 3353.4309082, -6153.3349609, 8494.5117188
4: -8050.6425781, 5230.7934570, -3765.1281738, 2475.0305176, -10525.6728516, 8995.9208984

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4062316, upper bound: 4552.2043651
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1943492, upper bound: 4552.2018617
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5978.6865234, 4786.7617188, -4267.7421875, 3427.3356934, -9406.0224609, 9054.5039062
1: -4799.3916016, 4603.5590820, -3433.8293457, 3298.9086914, -8098.3002930, 8037.3886719
2: -6875.3281250, 5000.4335938, -4925.3828125, 3575.4016113, -10450.7294922, 9925.8154297
3: -2637.8525391, 6785.8408203, -1881.0106201, 4863.0942383, -7495.9965820, 8664.7578125
4: -7630.0805664, 4929.4306641, -5456.5566406, 3519.1264648, -11149.2050781, 10385.9873047

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.0207390, upper bound: 4552.1888610
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9460957, upper bound: 4552.1867401
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6363.1420898, 5118.8481445, -4267.7421875, 3427.3356934, -9790.4775391, 9386.5888672
1: -5105.9799805, 4921.5893555, -3433.8293457, 3298.9086914, -8404.8886719, 8355.4189453
2: -7312.7739258, 5349.3676758, -4925.3828125, 3575.4016113, -10888.1757812, 10274.7490234
3: -2822.4138184, 7229.1621094, -1881.0106201, 4863.0942383, -7681.6367188, 9109.8525391
4: -8119.0649414, 5272.7631836, -5456.5566406, 3519.1264648, -11638.1914062, 10729.3203125

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4646321, upper bound: 4552.1882099
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844829, upper bound: 4552.1856417
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -5059.5898438, 4091.7714844, -8688.8886719, 8766.7421875
1: -3687.0437012, 3569.1484375, -4055.8063965, 3939.3488770, -7626.3911133, 7624.9550781
2: -5276.2031250, 3879.7956543, -5803.9467773, 4281.0913086, -9551.7285156, 9677.4492188
3: -2047.8087158, 5221.5805664, -2258.0759277, 5750.6269531, -7797.7163086, 7475.4555664
4: -5870.9716797, 3822.0715332, -6459.7363281, 4216.7597656, -10080.1718750, 10273.5908203

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9588421, upper bound: 4551.0222908
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9588421, upper bound: 4551.4667228
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -5059.5898438, 4091.7714844, -9195.7304688, 9182.8730469
1: -4091.2307129, 3969.2382812, -4055.8063965, 3939.3488770, -8030.5786133, 8025.0449219
2: -5853.0830078, 4314.3935547, -5803.9467773, 4281.0913086, -10129.8867188, 10113.6835938
3: -2275.4304199, 5795.1562500, -2258.0759277, 5750.6269531, -8026.0561523, 8050.4799805
4: -6514.6850586, 4250.0151367, -6459.7363281, 4216.7597656, -10725.1142578, 10702.7304688

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9635000, upper bound: 4551.0222908
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9635000, upper bound: 4551.4667228
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -5104.3461914, 4125.3242188, -8721.9003906, 8811.8896484
1: -3687.0437012, 3569.1484375, -4091.1743164, 3971.0102539, -7658.0522461, 7660.3227539
2: -5276.2031250, 3879.7956543, -5852.9707031, 4316.5053711, -9586.6015625, 9730.3574219
3: -2047.8087158, 5221.5805664, -2276.7976074, 5797.0576172, -7844.5283203, 7493.8945312
4: -5870.9716797, 3822.0715332, -6515.3515625, 4252.2592773, -10114.9843750, 10332.9179688

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9500306, upper bound: 4551.9476220
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9500306, upper bound: 4552.1865734
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -5104.3461914, 4125.3242188, -9229.2851562, 9227.6289062
1: -4091.2307129, 3969.2382812, -4091.1743164, 3971.0102539, -8062.2397461, 8060.4125977
2: -5853.0830078, 4314.3935547, -5852.9707031, 4316.5053711, -10164.7597656, 10166.5908203
3: -2275.4304199, 5795.1562500, -2276.7976074, 5797.0576172, -8072.4873047, 8068.9189453
4: -6514.6850586, 4250.0151367, -6515.3515625, 4252.2592773, -10759.9257812, 10762.0566406

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9508831, upper bound: 4551.9476220
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9508831, upper bound: 4552.1865734
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -6409.9711914, 5154.8813477, -5030.1420898, 4068.0788574, -10478.0498047, 10178.9746094
1: -5143.1816406, 4956.9326172, -4032.0764160, 3916.5642090, -9059.7460938, 8985.8652344
2: -7365.3452148, 5387.3491211, -5769.9741211, 4256.2983398, -11612.3935547, 11145.6132812
3: -2842.3461914, 7281.3403320, -2244.9921875, 5716.9531250, -8555.8964844, 9504.6464844
4: -8178.5961914, 5310.3564453, -6422.0834961, 4192.4262695, -12364.8427734, 11717.3320312

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9504688, upper bound: 4551.3990399
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1888560, upper bound: 4551.3978719
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -6411.2866211, 5155.9018555, -5074.3295898, 4101.2333984, -10512.5175781, 10227.7285156
1: -5144.2456055, 4957.9106445, -4067.0036621, 3947.8349609, -9092.0800781, 9024.4619141
2: -7366.9101562, 5388.3999023, -5818.3769531, 4291.3007812, -11648.3847656, 11198.9677734
3: -2842.9011230, 7282.8505859, -2263.4909668, 5762.7929688, -8602.6708984, 9524.3408203
4: -8180.2905273, 5311.3925781, -6477.0117188, 4227.5278320, -12400.9746094, 11777.0839844

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9481862, upper bound: 4552.1789576
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1865733, upper bound: 4552.1777896
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6366.1660156, 5119.2890625, -6465.6679688, 5197.9145508, -11557.8154297, 11577.7519531
1: -5110.7729492, 4923.2451172, -5188.6606445, 4998.1103516, -10105.2763672, 10107.7431641
2: -7322.4482422, 5348.0849609, -7431.6689453, 5431.6601562, -12735.1406250, 12760.1982422
3: -2820.9536133, 7239.8808594, -2865.7429199, 7345.2402344, -10138.3300781, 10080.5986328
4: -8126.7597656, 5270.9204102, -8250.3779297, 5354.0131836, -13465.0605469, 13504.7070312

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4646321, upper bound: 4551.4668377
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1872591
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6390.3413086, 5137.2963867, -6465.6679688, 5197.9145508, -11584.8183594, 11595.4892578
1: -5128.1879883, 4939.5961914, -5188.6606445, 4998.1103516, -10125.4355469, 10124.1347656
2: -7344.9951172, 5367.6391602, -7431.6689453, 5431.6601562, -12761.6904297, 12779.8447266
3: -2831.8232422, 7260.3583984, -2865.7429199, 7345.2402344, -10149.3730469, 10101.5341797
4: -8154.4711914, 5291.2568359, -8250.3779297, 5354.0131836, -13496.5107422, 13524.7314453

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0148684, upper bound: 4551.9476329
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844829, upper bound: 4552.1856363
time: 0.87 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 6.81 seconds
IS_A1_A1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4552.1691642, upper bound: 4552.1943617
IS_A1_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4552.1998742, upper bound: 4552.1943493
IS_A1_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4551.4369877, upper bound: 4552.1941249
IS_A1_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4551.4369877, upper bound: 4552.1917008
IS_A1_A1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4552.2051240, upper bound: 4551.0306479
IS_A1_A1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4552.2051240, upper bound: 4551.0306479
IS_A1_A1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4552.2043651, upper bound: 4551.4062316
IS_A1_A1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4552.2043651, upper bound: 4551.4744237
IS_A1_A1_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4552.2030297, upper bound: 4551.9559997
IS_A1_A1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4552.2030297, upper bound: 4551.9559997
IS_A1_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4552.0018232, upper bound: 4552.1910870
IS_A1_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4552.2002294, upper bound: 4552.1928830
IS_A1_A2_B1_B1_B1_A1, status: Status.VERIFIED, split count: 6, time: 6.81
Output dim: 0, lower bound: -4551.4744292, upper bound: 4551.4375556
IS_A1_A2_B1_B1_B1_A2, status: Status.VERIFIED, split count: 6, time: 6.81
Output dim: 0, lower bound: -4551.4744292, upper bound: 4551.4375556
IS_A1_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4551.4744292, upper bound: 4552.2003084
IS_A1_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4551.4744292, upper bound: 4552.2003084
IS_A1_A2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4552.1934080, upper bound: 4551.1882687
IS_A1_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4552.1888560, upper bound: 4551.3963185
IS_A1_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4552.1927693, upper bound: 4552.1513498
IS_A1_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4552.1865733, upper bound: 4552.1762363
IS_A1_A2_B2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.81
Output dim: 0, lower bound: -4551.4646324, upper bound: 4551.4630992
IS_A1_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1837414
IS_A1_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4552.1844829, upper bound: 4551.4630992
IS_A1_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4552.1827801, upper bound: 4552.1837414
IS_A1_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4552.1892986, upper bound: 4551.0207390
IS_A1_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4552.1885396, upper bound: 4551.4630992
IS_A1_A2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4552.1872042, upper bound: 4551.9460957
IS_A1_A2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4552.1843335, upper bound: 4552.1837216
IS_A2_B1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4551.1864078, upper bound: 4552.1893553
IS_A2_B1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4551.1870258, upper bound: 4552.1934081
IS_A2_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4552.1494888, upper bound: 4552.1885298
IS_A2_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4552.1501069, upper bound: 4552.1925717
IS_A2_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4551.3944576, upper bound: 4552.1843529
IS_A2_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4551.3950756, upper bound: 4552.1884367
IS_A2_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4552.1743754, upper bound: 4552.1817034
IS_A2_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4552.1749934, upper bound: 4552.1857872
IS_A2_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4551.0306479, upper bound: 4552.2051240
IS_A2_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4551.9559997, upper bound: 4552.2030297
IS_A2_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4551.4062316, upper bound: 4552.2043651
IS_A2_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4552.1943492, upper bound: 4552.2018617
IS_A2_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4551.0207390, upper bound: 4552.1888610
IS_A2_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4551.9460957, upper bound: 4552.1867401
IS_A2_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4551.4646321, upper bound: 4552.1882099
IS_A2_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4552.1844829, upper bound: 4552.1856417
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4551.9588421, upper bound: 4551.0222908
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4551.9588421, upper bound: 4551.4667228
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4551.9635000, upper bound: 4551.0222908
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4551.9635000, upper bound: 4551.4667228
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4551.9500306, upper bound: 4551.9476220
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4551.9500306, upper bound: 4552.1865734
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4551.9508831, upper bound: 4551.9476220
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4551.9508831, upper bound: 4552.1865734
IS_A2_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4551.9504688, upper bound: 4551.3990399
IS_A2_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4552.1888560, upper bound: 4551.3978719
IS_A2_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4551.9481862, upper bound: 4552.1789576
IS_A2_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4552.1865733, upper bound: 4552.1777896
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.81
Output dim: 0, lower bound: -4551.4646321, upper bound: 4551.4668377
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1872591
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4552.0148684, upper bound: 4551.9476329
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -4552.1844829, upper bound: 4552.1856363

## BFS IS instance: IS_A1_A1_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -2409.0625000, 1957.5953369, -2890.3549805, 2354.6845703, -4763.7465820, 4847.9497070
1: -1930.3334961, 1886.0590820, -2314.6940918, 2268.1701660, -4198.5039062, 4200.7514648
2: -2763.4660645, 2050.5446777, -3310.4218750, 2465.3471680, -5228.8129883, 5360.9667969
3: -1081.5762939, 2736.1838379, -1298.0891113, 3286.2832031, -4367.8593750, 4034.2724609
4: -3074.1376953, 2015.7597656, -3685.1674805, 2425.0095215, -5499.1469727, 5700.9272461

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B1_B2_B1_A1_A1

### Relational analysis result of IS_A1_A1_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.1878346, upper bound: 4552.1943617
time: 0.84 seconds

## Relational analysis of IS_A1_A1_B1_B2_B1_A1_A2

### Relational analysis result of IS_A1_A1_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.1878346, upper bound: 4552.1919229
time: 0.94 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -2898.2761230, 2358.8801270, -2890.3549805, 2354.6845703, -5252.9604492, 5249.2343750
1: -2321.5134277, 2272.3066406, -2314.6940918, 2268.1701660, -4589.6835938, 4586.9995117
2: -3320.6003418, 2471.0649414, -3310.4218750, 2465.3471680, -5785.9472656, 5781.4863281
3: -1302.0062256, 3290.8767090, -1298.0891113, 3286.2832031, -4588.2895508, 4588.9648438
4: -3695.3811035, 2429.6860352, -3685.1674805, 2425.0095215, -6120.3906250, 6114.8535156

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B1_B2_B1_A2_A1

### Relational analysis result of IS_A1_A1_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4365536, upper bound: 4552.1943493
time: 1.09 seconds

## Relational analysis of IS_A1_A1_B1_B2_B1_A2_A2

### Relational analysis result of IS_A1_A1_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4365536, upper bound: 4552.1918502
time: 0.76 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -2829.3090820, 2307.1660156, -2830.7468262, 2305.1364746, -5134.4453125, 5137.9130859
1: -2266.5866699, 2223.4016113, -2267.0932617, 2220.3234863, -4486.9096680, 4490.4941406
2: -3243.7138672, 2416.2275391, -3242.8298340, 2413.6757812, -5657.3896484, 5659.0571289
3: -1272.9768066, 3218.7609863, -1271.3063965, 3216.4094238, -4489.3857422, 4490.0673828
4: -3609.5827637, 2374.6940918, -3609.7048340, 2373.3562012, -5982.9379883, 5984.3989258

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_B2_B2_A1_B1

### Relational analysis result of IS_A1_A1_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4369877, upper bound: 4552.1926822
time: 0.73 seconds

## Relational analysis of IS_A1_A1_B1_B2_B2_A1_B2

### Relational analysis result of IS_A1_A1_B1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4369877, upper bound: 4552.1941249
time: 0.97 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -2854.9377441, 2325.1379395, -2830.7468262, 2305.1364746, -5160.0742188, 5155.8847656
1: -2286.3728027, 2239.5864258, -2267.0932617, 2220.3234863, -4506.6962891, 4506.6796875
2: -3270.2888184, 2435.0783691, -3242.8298340, 2413.6757812, -5683.9643555, 5677.9082031
3: -1282.8604736, 3243.4042969, -1271.3063965, 3216.4094238, -4499.2700195, 4514.7109375
4: -3640.6374512, 2394.5166016, -3609.7048340, 2373.3562012, -6013.9936523, 6004.2207031

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_B2_B2_A2_B1

### Relational analysis result of IS_A1_A1_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4369877, upper bound: 4552.1902856
time: 0.86 seconds

## Relational analysis of IS_A1_A1_B1_B2_B2_A2_B2

### Relational analysis result of IS_A1_A1_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4369877, upper bound: 4552.1917008
time: 0.78 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -4483.1152344, 3618.3911133, -6571.0732422, 6885.9169922
1: -2365.3845215, 2314.5366211, -3595.3923340, 3483.8129883, -5849.1972656, 5909.9287109
2: -3383.4587402, 2517.0407715, -5146.2998047, 3785.9094238, -7169.3681641, 7663.3393555
3: -1326.4201660, 3353.4309082, -1998.0097656, 5096.2939453, -6422.7138672, 5351.4404297
4: -3765.1281738, 2475.0305176, -5726.4765625, 3729.2534180, -7494.3808594, 8201.5068359

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B2_B1_B1_B1_A1

### Relational analysis result of IS_A1_A1_B2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0058751, upper bound: 4551.0265521
time: 0.84 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_B1_A2

### Relational analysis result of IS_A1_A1_B2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2042813, upper bound: 4551.0292738
time: 0.78 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -5830.6508789, 4671.1152344, -7623.7973633, 8233.4521484
1: -2365.3845215, 2314.5366211, -4681.3027344, 4493.1020508, -6858.4858398, 6995.8393555
2: -3383.4587402, 2517.0407715, -6707.6035156, 4879.2382812, -8262.6953125, 9224.6435547
3: -1326.4201660, 3353.4309082, -2573.3769531, 6623.4487305, -7949.7460938, 5926.8071289
4: -3765.1281738, 2475.0305176, -7443.5112305, 4809.1323242, -8574.2578125, 9918.5390625

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B2_B1_B1_B2_A1

### Relational analysis result of IS_A1_A1_B2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0058751, upper bound: 4551.0265521
time: 0.76 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_B2_A2

### Relational analysis result of IS_A1_A1_B2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2042813, upper bound: 4551.0292738
time: 0.83 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -4976.5166016, 4024.3906250, -6977.0727539, 7379.3183594
1: -2365.3845215, 2314.5366211, -3988.8959961, 3874.3952637, -6239.7797852, 6303.4326172
2: -3383.4587402, 2517.0407715, -5707.9775391, 4210.3144531, -7593.7729492, 8225.0175781
3: -1326.4201660, 3353.4309082, -2220.2397461, 5655.1889648, -6981.6088867, 5573.6708984
4: -3765.1281738, 2475.0305176, -6353.2714844, 4147.1865234, -7912.3134766, 8828.3007812

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B2_B1_B2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0046451, upper bound: 4551.4029169
time: 0.78 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2_B1_A2

### Relational analysis result of IS_A1_A1_B2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2030513, upper bound: 4551.4048419
time: 0.84 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -6193.8574219, 4986.5454102, -7939.2275391, 8596.6591797
1: -2365.3845215, 2314.5366211, -4970.9995117, 4795.1953125, -7160.5800781, 7285.5361328
2: -3383.4587402, 2517.0407715, -7121.1513672, 5210.3613281, -8593.8183594, 9638.1923828
3: -1326.4201660, 3353.4309082, -2748.6206055, 7044.2285156, -8370.6484375, 6102.0517578
4: -3765.1281738, 2475.0305176, -7905.8334961, 5135.0805664, -8900.2089844, 10380.8642578

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B2_B1_B2_B2_A1

### Relational analysis result of IS_A1_A1_B2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0046451, upper bound: 4551.4712998
time: 0.87 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2_B2_A2

### Relational analysis result of IS_A1_A1_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2030513, upper bound: 4551.4728823
time: 0.82 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -4514.2919922, 3641.8427734, -6594.5249023, 6917.0937500
1: -2365.3845215, 2314.5366211, -3620.1120605, 3505.6416016, -5871.0263672, 5934.6484375
2: -3383.4587402, 2517.0407715, -5180.1918945, 3810.8364258, -7194.2949219, 7697.2309570
3: -1326.4201660, 3353.4309082, -2011.2974854, 5128.1811523, -6454.6005859, 5364.7285156
4: -3765.1281738, 2475.0305176, -5765.2221680, 3754.4975586, -7519.6250000, 8240.2519531

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B2_B2_B1_B1_A1

### Relational analysis result of IS_A1_A1_B2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0037493, upper bound: 4551.9518618
time: 0.87 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_B1_A2

### Relational analysis result of IS_A1_A1_B2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2021555, upper bound: 4551.9546184
time: 0.97 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -5837.9506836, 4675.3857422, -7628.0678711, 8240.7519531
1: -2365.3845215, 2314.5366211, -4685.7221680, 4496.4042969, -6861.7890625, 7000.2587891
2: -3383.4587402, 2517.0407715, -6711.3725586, 4883.9682617, -8267.4267578, 9228.4121094
3: -1326.4201660, 3353.4309082, -2576.2990723, 6626.1103516, -7952.5302734, 5929.7299805
4: -3765.1281738, 2475.0305176, -7450.1875000, 4815.0356445, -8580.1621094, 9925.2177734

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B2_B2_B1_B2_A1

### Relational analysis result of IS_A1_A1_B2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0037493, upper bound: 4551.9518618
time: 0.77 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_B2_A2

### Relational analysis result of IS_A1_A1_B2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2021555, upper bound: 4551.9546184
time: 0.77 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -2643.8227539, 2149.3374023, -5031.4194336, 4066.2902832, -6710.1123047, 7180.7563477
1: -2117.3525391, 2070.5566406, -4032.4670410, 3914.1293945, -6031.4819336, 6103.0234375
2: -3028.8117676, 2251.8281250, -5768.8647461, 4254.6137695, -7283.4252930, 8020.6918945
3: -1188.8460693, 2998.9645996, -2243.8178711, 5713.5878906, -6902.4335938, 5242.7822266
4: -3371.2492676, 2214.3798828, -6422.0249023, 4191.3217773, -7562.5708008, 8636.4042969

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B2_B2_A1_B1

### Relational analysis result of IS_A1_A1_B2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0001364, upper bound: 4552.1828046
time: 0.85 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_A1_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.0018232, upper bound: 4552.1910870
time: 0.83 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -2898.7780762, 2359.8454590, -5049.5405273, 4080.6557617, -6979.4335938, 7409.3857422
1: -2322.1835938, 2272.8837891, -4047.0185547, 3927.9313965, -6250.1147461, 6319.9023438
2: -3321.8449707, 2471.8774414, -5789.6328125, 4269.5668945, -7591.4116211, 8261.5097656
3: -1302.9533691, 3291.8579102, -2251.6179199, 5734.0175781, -7036.9702148, 5543.4755859
4: -3696.7353516, 2430.7426758, -6445.0820312, 4206.0708008, -7902.8041992, 8875.8232422

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B2_B2_A2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2002294, upper bound: 4552.1847609
time: 0.82 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_A2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2002294, upper bound: 4552.1928830
time: 1.03 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -4091.7839355, 3287.7749023, -2854.9377441, 2325.1379395, -6416.9218750, 6142.7128906
1: -3292.9262695, 3166.0788574, -2286.3728027, 2239.5864258, -5532.5126953, 5452.4516602
2: -4724.7177734, 3429.6672363, -3270.2888184, 2435.0783691, -7159.7958984, 6699.9560547
3: -1803.3294678, 4667.9692383, -1282.8604736, 3243.4042969, -5046.7333984, 5950.8295898
4: -5233.6342773, 3374.2546387, -3640.6374512, 2394.5166016, -7628.1494141, 7014.8920898

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_A1

### Relational analysis result of IS_A1_A2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4725785, upper bound: 4552.1984474
time: 0.80 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_A2

### Relational analysis result of IS_A1_A2_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4732232, upper bound: 4552.1990655
time: 0.78 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -4104.0507812, 3298.7883301, -2854.9377441, 2325.1379395, -6429.1884766, 6153.7255859
1: -3300.4887695, 3174.7365723, -2286.3728027, 2239.5864258, -5540.0751953, 5461.1093750
2: -4733.4013672, 3441.2429199, -3270.2888184, 2435.0783691, -7168.4794922, 6711.5312500
3: -1810.9291992, 4675.3427734, -1282.8604736, 3243.4042969, -5054.3334961, 5958.2031250
4: -5245.8227539, 3387.6132812, -3640.6374512, 2394.5166016, -7640.3388672, 7028.2509766

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A1

### Relational analysis result of IS_A1_A2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4725785, upper bound: 4552.1944551
time: 0.83 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A2

### Relational analysis result of IS_A1_A2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4732232, upper bound: 4552.1950678
time: 0.81 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -4168.9941406, 3352.2631836, -4483.1152344, 3618.3911133, -7787.3852539, 7835.3784180
1: -3352.3190918, 3226.2553711, -3595.3923340, 3483.8129883, -6836.1313477, 6821.6469727
2: -4807.1967773, 3498.0939941, -5146.2998047, 3785.9094238, -8593.1044922, 8644.3925781
3: -1841.5876465, 4747.8769531, -1998.0097656, 5096.2939453, -6937.8803711, 6745.0942383
4: -5327.3793945, 3444.0600586, -5726.4765625, 3729.2534180, -9056.6328125, 9170.5361328

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A2_B1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_B1_B2_B1_B1_A1

### Relational analysis result of IS_A1_A2_B1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1893553, upper bound: 4551.1864078
time: 1.14 seconds

## Relational analysis of IS_A1_A2_B1_B2_B1_B1_A2

### Relational analysis result of IS_A1_A2_B1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1934080, upper bound: 4551.1870258
time: 0.95 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -4168.2558594, 3351.7019043, -4976.5166016, 4024.3906250, -8192.6455078, 8328.2177734
1: -3351.7114258, 3225.7104492, -3988.8959961, 3874.3952637, -7226.1064453, 7214.6054688
2: -4806.3129883, 3497.5136719, -5707.9775391, 4210.3144531, -9016.6269531, 9205.4902344
3: -1841.2913818, 4747.0151367, -2220.2397461, 5655.1889648, -7496.4804688, 6967.2548828
4: -5326.4140625, 3443.4963379, -6353.2714844, 4147.1865234, -9473.6005859, 9796.7636719

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A2_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_B1_B2_B1_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1843529, upper bound: 4551.3944576
time: 0.94 seconds

## Relational analysis of IS_A1_A2_B1_B2_B1_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1884366, upper bound: 4551.3950756
time: 0.83 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -4170.5756836, 3353.4689941, -4514.2919922, 3641.8427734, -7812.4184570, 7867.7597656
1: -3353.6210938, 3227.4238281, -3620.1120605, 3505.6416016, -6859.2626953, 6847.5351562
2: -4809.0878906, 3499.3376465, -5180.1918945, 3810.8364258, -8619.9238281, 8679.5283203
3: -1842.2222900, 4749.7207031, -2011.2974854, 5128.1811523, -6970.4023438, 6759.8217773
4: -5329.4477539, 3445.2673340, -5765.2221680, 3754.4975586, -9083.9453125, 9210.4863281

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_B1_B2_B2_B1_A1

### Relational analysis result of IS_A1_A2_B1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1885298, upper bound: 4552.1494888
time: 0.84 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2_B1_A2

### Relational analysis result of IS_A1_A2_B1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1925716, upper bound: 4552.1501069
time: 0.85 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -4170.1416016, 3353.1374512, -5020.0449219, 4056.9860840, -8227.1269531, 8373.1816406
1: -3353.2631836, 3227.1027832, -4023.2673340, 3905.1572266, -7258.4204102, 7250.3696289
2: -4808.5683594, 3498.9963379, -5755.6445312, 4244.7993164, -9053.3662109, 9254.6386719
3: -1842.0480957, 4749.2148438, -2238.5415039, 5700.3437500, -7542.3896484, 6987.7563477
4: -5328.8793945, 3444.9353027, -6407.4130859, 4181.7666016, -9510.6425781, 9852.3486328

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_B1_B2_B2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1817034, upper bound: 4552.1743754
time: 0.84 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1857871, upper bound: 4552.1749934
time: 0.84 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -4175.5634766, 3350.7539062, -4186.9238281, 3361.6271973, -7537.1904297, 7537.6777344
1: -3362.1945801, 3227.0385742, -3369.0449219, 3235.5971680, -6597.7915039, 6596.0834961
2: -4825.2534180, 3494.4333496, -4832.7299805, 3506.0024414, -8331.2558594, 8327.1630859
3: -1836.1400146, 4765.6860352, -1843.7967529, 4772.0375977, -6608.1777344, 6609.4819336
4: -5343.5141602, 3436.8771973, -5354.3334961, 3450.2895508, -8793.8007812, 8791.2109375

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_A1

### Relational analysis result of IS_A1_A2_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4550.4644370, upper bound: 4552.1871893
time: 0.84 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_A2

### Relational analysis result of IS_A1_A2_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.3945056, upper bound: 4552.1859805
time: 0.78 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4186.9238281, 3361.6271973, -4175.5634766, 3350.7539062, -7537.6777344, 7537.1904297
1: -3369.0449219, 3235.5971680, -3362.1945801, 3227.0385742, -6596.0834961, 6597.7910156
2: -4832.7299805, 3506.0024414, -4825.2534180, 3494.4333496, -8327.1630859, 8331.2558594
3: -1843.7967529, 4772.0375977, -1836.1400146, 4765.6860352, -6609.4824219, 6608.1777344
4: -5354.3334961, 3450.2895508, -5343.5141602, 3436.8771973, -8791.2109375, 8793.8007812

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A2_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_A2_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_B2_B1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1826219, upper bound: 4551.4611806
time: 0.88 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_B1_B2

### Relational analysis result of IS_A1_A2_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1832401, upper bound: 4551.4617931
time: 0.79 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4186.9238281, 3361.6271973, -4186.9238281, 3361.6271973, -7548.5507812, 7548.5507812
1: -3369.0449219, 3235.5971680, -3369.0449219, 3235.5971680, -6604.6420898, 6604.6420898
2: -4832.7299805, 3506.0024414, -4832.7299805, 3506.0024414, -8338.7324219, 8338.7324219
3: -1843.7967529, 4772.0375977, -1843.7967529, 4772.0375977, -6615.8344727, 6615.8344727
4: -5354.3334961, 3450.2895508, -5354.3334961, 3450.2895508, -8804.6230469, 8804.6230469

Time for backsubstitution: 1.67 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=5687.5751953125
rel_dist={0: [-4552.287863642099, 4552.287863642101]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2821309, upper bound: 4552.2729710
time: 1.08 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2717824, upper bound: 4552.2717824
time: 0.89 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.12 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.12
Output dim: 0, lower bound: -4552.2821309, upper bound: 4552.2729710
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.12
Output dim: 0, lower bound: -4552.2717824, upper bound: 4552.2717824

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -2987.9899902, 2430.9243164, -3136.6748047, 2550.9006348, -5538.8906250, 5567.5991211
1: -2393.8061523, 2341.6025391, -2513.2941895, 2457.4460449, -4851.2519531, 4854.8964844
2: -3424.2626953, 2546.4648438, -3595.4755859, 2672.1398926, -6096.4023438, 6141.9404297
3: -1341.7878418, 3393.8977051, -1407.5072021, 3564.2558594, -4906.0439453, 4801.4047852
4: -3810.2302246, 2503.8750000, -3999.4067383, 2627.0041504, -6437.2343750, 6503.2817383

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2714855, upper bound: 4552.2714855
time: 0.77 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2714855, upper bound: 4552.2714855
time: 0.76 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -5188.6577148, 4191.8315430, -3086.8498535, 2511.7026367, -7700.3603516, 7278.6816406
1: -4159.4165039, 4035.3022461, -2473.1953125, 2419.7272949, -6579.1435547, 6508.4975586
2: -5950.7929688, 4386.3300781, -3537.9782715, 2631.1450195, -8581.9365234, 7924.3076172
3: -2313.8149414, 5892.1474609, -1385.9432373, 3507.9636230, -5821.7783203, 7278.0898438
4: -6623.0805664, 4320.7236328, -3935.8896484, 2586.6906738, -9209.7705078, 8256.6132812

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2714855, upper bound: 4552.2717824
time: 0.74 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2714855, upper bound: 4552.2717824
time: 0.81 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.15 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 0, lower bound: -4552.2714855, upper bound: 4552.2714855
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 0, lower bound: -4552.2714855, upper bound: 4552.2714855
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 0, lower bound: -4552.2714855, upper bound: 4552.2717824
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 0, lower bound: -4552.2714855, upper bound: 4552.2717824

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -2987.9899902, 2430.9243164, -2987.9899902, 2430.9243164, -5418.9140625, 5418.9140625
1: -2393.8061523, 2341.6025391, -2393.8061523, 2341.6025391, -4735.4086914, 4735.4086914
2: -3424.2626953, 2546.4648438, -3424.2626953, 2546.4648438, -5970.7275391, 5970.7275391
3: -1341.7878418, 3393.8977051, -1341.7878418, 3393.8977051, -4735.6855469, 4735.6855469
4: -3810.2302246, 2503.8750000, -3810.2302246, 2503.8750000, -6314.1054688, 6314.1054688

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1901063, upper bound: 4552.1892355
time: 0.84 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1866835, upper bound: 4552.1866834
time: 0.87 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -2987.9899902, 2430.9243164, -5187.8886719, 4191.2221680, -7179.2119141, 7618.8129883
1: -2393.8061523, 2341.6025391, -4158.8027344, 4034.7153320, -6428.5205078, 6500.4052734
2: -3424.2626953, 2546.4648438, -5949.9160156, 4385.6933594, -7809.9560547, 8496.3798828
3: -1341.7878418, 3393.8977051, -2313.4826660, 5891.2778320, -7233.0654297, 5707.3803711
4: -3810.2302246, 2503.8750000, -6622.1035156, 4320.0986328, -8130.3276367, 9125.9785156

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1901063, upper bound: 4552.1892355
time: 0.84 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1866835, upper bound: 4552.1866834
time: 0.87 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -5187.8886719, 4191.2221680, -2979.2597656, 2424.2385254, -7612.1264648, 7170.4819336
1: -4158.8027344, 4034.7153320, -2386.7011719, 2335.1318359, -6493.9345703, 6421.4160156
2: -5949.9160156, 4385.6933594, -3414.0134277, 2539.3676758, -8489.2832031, 7799.7070312
3: -2313.4826660, 5891.2778320, -1337.9373779, 3384.0065918, -5697.4892578, 7229.2153320
4: -6622.1035156, 4320.0986328, -3798.9916992, 2496.9252930, -9119.0283203, 8119.0903320

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1892355, upper bound: 4552.1904055
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1843458, upper bound: 4552.1889483
time: 0.84 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -5188.6577148, 4191.8315430, -5188.6577148, 4191.8315430, -9380.4892578, 9380.4892578
1: -4159.4165039, 4035.3022461, -4159.4165039, 4035.3022461, -8194.7187500, 8194.7187500
2: -5950.7929688, 4386.3300781, -5950.7929688, 4386.3300781, -10334.5771484, 10334.5771484
3: -2313.8149414, 5892.1474609, -2313.8149414, 5892.1474609, -8205.9628906, 8205.9628906
4: -6623.0805664, 4320.7236328, -6623.0805664, 4320.7236328, -10938.7910156, 10938.7919922

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1892355, upper bound: 4552.1904055
time: 1.96 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1866834, upper bound: 4552.1889483
time: 0.96 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.55 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 4.55
Output dim: 0, lower bound: -4552.1901063, upper bound: 4552.1892355
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 4.55
Output dim: 0, lower bound: -4552.1866835, upper bound: 4552.1866834
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 4.55
Output dim: 0, lower bound: -4552.1901063, upper bound: 4552.1892355
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 4.55
Output dim: 0, lower bound: -4552.1866835, upper bound: 4552.1866834
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.55
Output dim: 0, lower bound: -4552.1892355, upper bound: 4552.1904055
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.55
Output dim: 0, lower bound: -4552.1843458, upper bound: 4552.1889483
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.55
Output dim: 0, lower bound: -4552.1892355, upper bound: 4552.1904055
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.55
Output dim: 0, lower bound: -4552.1866834, upper bound: 4552.1889483

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -2987.9899902, 2430.9243164, -2952.6821289, 2402.8015137, -5390.7915039, 5383.6064453
1: -2393.8061523, 2341.6025391, -2365.3845215, 2314.5366211, -4708.3427734, 4706.9873047
2: -3424.2626953, 2546.4648438, -3383.4587402, 2517.0407715, -5941.3027344, 5929.9223633
3: -1341.7878418, 3393.8977051, -1326.4201660, 3353.4309082, -4695.2187500, 4720.3173828
4: -3810.2302246, 2503.8750000, -3765.1281738, 2475.0305176, -6285.2602539, 6269.0029297

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4690262, upper bound: 4552.2022458
time: 0.74 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1892226, upper bound: 4552.1987525
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -2954.7021484, 2403.8925781, -4184.7089844, 3364.2209473, -6318.9228516, 6588.6005859
1: -2367.0505371, 2315.4003906, -3365.2619629, 3237.8483887, -5604.8984375, 5680.6621094
2: -3385.9895020, 2517.9428711, -4825.9780273, 3510.4304199, -6896.4199219, 7343.9208984
3: -1326.8588867, 3355.8562012, -1847.8853760, 4766.1914062, -6093.0502930, 5203.7416992
4: -3767.7563477, 2476.0551758, -5347.9155273, 3456.0410156, -7223.7973633, 7823.9707031

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1846192, upper bound: 4552.1866835
time: 0.75 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1846192, upper bound: 4552.1866835
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -2987.9899902, 2430.9243164, -5157.1284180, 4166.5312500, -7154.5209961, 7588.0527344
1: -2393.8061523, 2341.6025391, -4134.0346680, 4010.9514160, -6404.7573242, 6475.6372070
2: -3424.2626953, 2546.4648438, -5914.4672852, 4359.8647461, -7784.1274414, 8460.9306641
3: -1341.7878418, 3393.8977051, -2299.8535156, 5856.1523438, -7197.9404297, 5693.7509766
4: -3810.2302246, 2503.8750000, -6582.8134766, 4294.7578125, -8104.9868164, 9086.6884766

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1899945, upper bound: 4551.3949576
time: 0.89 seconds

## Relational analysis of IS_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1883855, upper bound: 4552.1746604
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -2954.7021484, 2403.8925781, -6414.0766602, 5158.0693359, -8112.7714844, 8817.9677734
1: -2367.0505371, 2315.4003906, -5146.5043945, 4959.9829102, -7327.0332031, 7461.9047852
2: -3385.9895020, 2517.9428711, -7370.2324219, 5390.6318359, -8776.6210938, 9888.1728516
3: -1326.8588867, 3355.8562012, -2844.0791016, 7286.0581055, -8612.9169922, 6199.9355469
4: -3767.7563477, 2476.0551758, -8183.8876953, 5313.5903320, -9081.3466797, 10659.9433594

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1868994, upper bound: 4552.1866835
time: 0.85 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1868994, upper bound: 4552.1866835
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5157.1284180, 4166.5312500, -2979.2597656, 2424.2385254, -7581.3662109, 7145.7900391
1: -4134.0346680, 4010.9514160, -2386.7011719, 2335.1318359, -6469.1665039, 6397.6523438
2: -5914.4672852, 4359.8647461, -3414.0134277, 2539.3676758, -8453.8339844, 7773.8779297
3: -2299.8535156, 5856.1523438, -1337.9373779, 3384.0065918, -5683.8598633, 7194.0898438
4: -6582.8134766, 4294.7578125, -3798.9916992, 2496.9252930, -9079.7382812, 8093.7490234

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.3949576, upper bound: 4552.1899945
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1746604, upper bound: 4552.1883855
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6414.0766602, 5158.0693359, -2949.4614258, 2399.8703613, -8813.9472656, 8107.5307617
1: -5146.5043945, 4959.9829102, -2362.7854004, 2311.5070801, -7458.0117188, 7322.7680664
2: -7370.2329102, 5390.6318359, -3379.8408203, 2513.6713867, -9883.9013672, 8770.4726562
3: -2844.0791016, 7286.0581055, -1324.5412598, 3349.9157715, -6193.9946289, 8610.5996094
4: -8183.8876953, 5313.5903320, -3761.0117188, 2471.8740234, -10655.7617188, 9074.6015625

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1866835, upper bound: 4552.1868994
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1866835, upper bound: 4552.1889483
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -5188.6577148, 4191.8315430, -9350.2294922, 9356.1972656
1: -4135.0488281, 4011.9240723, -4159.4165039, 4035.3022461, -8170.3510742, 8171.3408203
2: -5915.9174805, 4360.9184570, -5950.7929688, 4386.3300781, -10299.6093750, 10309.1445312
3: -2300.4028320, 5857.5903320, -2313.8149414, 5892.1474609, -8192.5507812, 8171.4052734
4: -6584.4262695, 4295.7910156, -6623.0805664, 4320.7236328, -10900.1123047, 10913.8359375

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1843458, upper bound: 4552.1865694
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1843458, upper bound: 4552.1889483
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6411.8666992, 5156.3530273, -5155.3593750, 4164.8334961, -10576.6982422, 10308.0683594
1: -5144.7148438, 4958.3403320, -4132.6069336, 4009.2773438, -9153.9912109, 9089.6171875
2: -7367.6010742, 5388.8642578, -5912.4819336, 4358.1196289, -11717.7822266, 11292.0576172
3: -2843.1455078, 7283.5170898, -2299.0849609, 5854.1420898, -8693.7363281, 9562.7958984
4: -8181.0380859, 5311.8496094, -6580.6372070, 4293.0454102, -12469.0107422, 11879.8125000

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1866835, upper bound: 4552.1865694
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1866835, upper bound: 4552.1889483
time: 0.82 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.25 seconds
IS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -4551.4690262, upper bound: 4552.2022458
IS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -4552.1892226, upper bound: 4552.1987525
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -4552.1846192, upper bound: 4552.1866835
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -4552.1846192, upper bound: 4552.1866835
IS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -4552.1899945, upper bound: 4551.3949576
IS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -4552.1883855, upper bound: 4552.1746604
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -4552.1868994, upper bound: 4552.1866835
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -4552.1868994, upper bound: 4552.1866835
IS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -4551.3949576, upper bound: 4552.1899945
IS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -4552.1746604, upper bound: 4552.1883855
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -4552.1866835, upper bound: 4552.1868994
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -4552.1866835, upper bound: 4552.1889483
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -4552.1843458, upper bound: 4552.1865694
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -4552.1843458, upper bound: 4552.1889483
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -4552.1866835, upper bound: 4552.1865694
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -4552.1866835, upper bound: 4552.1889483

## BFS IS instance: IS_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -2864.9853516, 2335.5559082, -2932.8112793, 2387.1535645, -5252.1386719, 5268.3671875
1: -2295.3029785, 2250.7009277, -2349.4230957, 2299.4885254, -4594.7900391, 4600.1235352
2: -3284.9338379, 2445.8928223, -3360.7331543, 2500.5988770, -5785.5327148, 5806.6259766
3: -1288.4378662, 3259.7199707, -1317.7845459, 3331.2534180, -4619.6914062, 4577.5043945
4: -3655.1245117, 2403.8120117, -3739.9848633, 2458.7822266, -6113.9067383, 6143.7968750

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4690262, upper bound: 4551.4360878
time: 0.82 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4690262, upper bound: 4552.1987525
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -2890.1743164, 2353.1525879, -2952.6821289, 2402.8015137, -5292.9755859, 5305.8344727
1: -2314.7424316, 2266.5371094, -2365.3845215, 2314.5366211, -4629.2792969, 4631.9218750
2: -3311.0268555, 2464.3937988, -3383.4587402, 2517.0407715, -5828.0668945, 5847.8510742
3: -1298.1706543, 3283.7622070, -1326.4201660, 3353.4309082, -4651.6010742, 4610.1811523
4: -3685.6552734, 2423.2575684, -3765.1281738, 2475.0305176, -6160.6855469, 6188.3857422

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B1_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1892226, upper bound: 4551.4360878
time: 0.81 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2

### Relational analysis result of IS_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1892226, upper bound: 4552.1987525
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -4164.1435547, 3348.5708008, -6301.2524414, 6566.9453125
1: -2365.3845215, 2314.5366211, -3348.3281250, 3222.6748047, -5588.0595703, 5662.8647461
2: -3383.4587402, 2517.0407715, -4801.3999023, 3494.2836914, -6877.7416992, 7318.4399414
3: -1326.4201660, 3353.4309082, -1839.6420898, 4742.2231445, -6068.6425781, 5193.0732422
4: -3765.1281738, 2475.0305176, -5321.0410156, 3440.3591309, -7205.4868164, 7796.0712891

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4360878, upper bound: 4552.1862971
time: 0.91 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1811339, upper bound: 4552.1828398
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -4267.7421875, 3427.3356934, -4267.7421875, 3427.3356934, -7695.0771484, 7695.0771484
1: -3433.8293457, 3298.9086914, -3433.8293457, 3298.9086914, -6732.7377930, 6732.7377930
2: -4925.3828125, 3575.4016113, -4925.3828125, 3575.4016113, -8500.7832031, 8500.7841797
3: -1881.0106201, 4863.0942383, -1881.0106201, 4863.0942383, -6744.1049805, 6744.1049805
4: -5456.5566406, 3519.1264648, -5456.5566406, 3519.1264648, -8975.6835938, 8975.6835938

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

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
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1846100, upper bound: 4551.4614286
time: 0.85 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1811339, upper bound: 4552.1820814
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -2968.2436523, 2415.3955078, -5026.7231445, 4065.3764648, -7033.6201172, 7442.1181641
1: -2377.9462891, 2326.6662598, -4029.3537598, 3913.9570312, -6291.9033203, 6356.0195312
2: -3401.6904297, 2530.1479492, -5766.0830078, 4253.4780273, -7655.1684570, 8296.2304688
3: -1333.2133789, 3371.8791504, -2243.5178223, 5713.1000977, -7046.3134766, 5615.3964844
4: -3785.2519531, 2487.7487793, -6417.7583008, 4189.6606445, -7974.9125977, 8905.5058594

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B1_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1896128, upper bound: 4551.1871985
time: 0.74 seconds

## Relational analysis of IS_A1_B2_B1_B1_B2

### Relational analysis result of IS_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1874217, upper bound: 4551.3949576
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -2987.9899902, 2430.9243164, -5072.5488281, 4099.8198242, -7087.8090820, 7503.4731445
1: -2393.8061523, 2341.6025391, -4065.5808105, 3946.4719238, -6340.2783203, 6407.1835938
2: -3424.2626953, 2546.4648438, -5816.3437500, 4289.8242188, -7714.0869141, 8362.8076172
3: -1341.7878418, 3393.8977051, -2262.7209473, 5760.7783203, -7102.5664062, 5656.6186523
4: -3810.2302246, 2503.8750000, -6474.7514648, 4226.0800781, -8036.3090820, 8978.6259766

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B1_B2_B1

### Relational analysis result of IS_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1878615, upper bound: 4552.1500239
time: 0.82 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2

### Relational analysis result of IS_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1849632, upper bound: 4552.1746604
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -6398.4824219, 5145.9277344, -8098.6098633, 8801.2841797
1: -2365.3845215, 2314.5366211, -5133.8969727, 4948.3686523, -7313.7529297, 7448.4335938
2: -3383.4587402, 2517.0407715, -7351.6796875, 5378.1259766, -8761.5839844, 9868.7197266
3: -1326.4201660, 3353.4309082, -2837.4758301, 7268.1264648, -8594.5468750, 6190.9067383
4: -3765.1281738, 2475.0305176, -8163.7880859, 5301.2705078, -9066.3984375, 10638.8173828

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1862018, upper bound: 4551.9492265
time: 0.82 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1855638, upper bound: 4552.1866626
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -4267.7421875, 3427.3356934, -6465.6679688, 5197.9145508, -9465.6562500, 9893.0039062
1: -3433.8293457, 3298.9086914, -5188.6606445, 4998.1103516, -8431.9394531, 8487.5683594
2: -4925.3828125, 3575.4016113, -7431.6689453, 5431.6601562, -10357.0410156, 11007.0703125
3: -1881.0106201, 4863.0942383, -2865.7429199, 7345.2402344, -9226.2509766, 7724.6606445
4: -5456.5566406, 3519.1264648, -8250.3779297, 5354.0131836, -10810.5673828, 11769.5009766

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1862018, upper bound: 4551.9492265
time: 0.79 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1855638, upper bound: 4552.1862412
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -5026.7231445, 4065.3764648, -2959.4863281, 2408.6767578, -7435.3999023, 7024.8627930
1: -4029.3537598, 3913.9570312, -2370.8200684, 2320.1647949, -6349.5180664, 6284.7758789
2: -5766.0830078, 4253.4780273, -3391.4108887, 2523.0122070, -8289.0957031, 7644.8881836
3: -2243.5178223, 5713.1000977, -1329.3392334, 3361.9567871, -5605.4746094, 7042.4394531
4: -6417.7583008, 4189.6606445, -3773.9790039, 2480.7626953, -8898.5205078, 7963.6396484

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.1871985, upper bound: 4552.1896128
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_A1_A2

### Relational analysis result of IS_A2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.3949576, upper bound: 4552.1874217
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -5072.5488281, 4099.8198242, -2979.2597656, 2424.2385254, -7496.7866211, 7079.0786133
1: -4065.5808105, 3946.4719238, -2386.7011719, 2335.1318359, -6400.7128906, 6333.1728516
2: -5816.3437500, 4289.8242188, -3414.0134277, 2539.3676758, -8355.7109375, 7703.8378906
3: -2262.7209473, 5760.7783203, -1337.9373779, 3384.0065918, -5646.7275391, 7098.7158203
4: -6474.7514648, 4226.0800781, -3798.9916992, 2496.9252930, -8971.6757812, 8025.0717773

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_A2_A1

### Relational analysis result of IS_A2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1500239, upper bound: 4552.1880920
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A1_A2_A2

### Relational analysis result of IS_A2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1746604, upper bound: 4552.1849632
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6398.4824219, 5145.9277344, -2946.5974121, 2398.1357422, -8796.6181641, 8092.5253906
1: -5133.8969727, 4948.3686523, -2360.4313965, 2310.0227051, -7443.9194336, 7308.7998047
2: -7351.6796875, 5378.1259766, -3376.3110352, 2512.0881348, -9863.7675781, 8754.4365234
3: -2837.4758301, 7268.1264648, -1323.7344971, 3346.5322266, -6184.0073242, 8591.8603516
4: -8163.7880859, 5301.2705078, -3757.2946777, 2470.1801758, -10633.9677734, 9058.5644531

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9492265, upper bound: 4552.1862018
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1866626, upper bound: 4552.1855638
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6465.6679688, 5197.9145508, -4267.7421875, 3427.3356934, -9893.0039062, 9465.6562500
1: -5188.6606445, 4998.1103516, -3433.8293457, 3298.9086914, -8487.5683594, 8431.9394531
2: -7431.6689453, 5431.6601562, -4925.3828125, 3575.4016113, -11007.0703125, 10357.0410156
3: -2865.7429199, 7345.2402344, -1881.0106201, 4863.0942383, -7724.6611328, 9226.2509766
4: -8250.3779297, 5354.0131836, -5456.5566406, 3519.1264648, -11769.5009766, 10810.5673828

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9492265, upper bound: 4552.1878615
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1866626, upper bound: 4552.1874041
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -5158.3984375, 4167.5395508, -9325.9375000, 9325.9375000
1: -4135.0488281, 4011.9240723, -4135.0488281, 4011.9240723, -8146.9726562, 8146.9726562
2: -5915.9174805, 4360.9184570, -5915.9174805, 4360.9184570, -10274.1767578, 10274.1767578
3: -2300.4028320, 5857.5903320, -2300.4028320, 5857.5903320, -8157.9863281, 8157.9863281
4: -6584.4262695, 4295.7910156, -6584.4262695, 4295.7910156, -10875.1572266, 10875.1562500

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.3949576, upper bound: 4552.1819804
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1746604, upper bound: 4552.1804615
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -6395.8935547, 5143.9062500, -10298.7148438, 10563.4335938
1: -4135.0488281, 4011.9240723, -5131.8076172, 4946.4331055, -9080.1894531, 9143.7314453
2: -5915.9174805, 4360.9184570, -7348.6030273, 5376.0405273, -11282.7089844, 11701.9794922
3: -2300.4028320, 5857.5903320, -2836.3747559, 7265.1459961, -9546.0849609, 8690.4970703
4: -6584.4262695, 4295.7910156, -8160.4511719, 5299.2163086, -11871.0273438, 12451.3417969

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.3949576, upper bound: 4552.1899945
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1746604, upper bound: 4552.1883855
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6395.8935547, 5143.9062500, -5158.3984375, 4167.5395508, -10563.4335938, 10298.7148438
1: -5131.8076172, 4946.4331055, -4135.0488281, 4011.9240723, -9143.7314453, 9080.1894531
2: -7348.6030273, 5376.0405273, -5915.9174805, 4360.9184570, -11701.9794922, 11282.7089844
3: -2836.3747559, 7265.1459961, -2300.4028320, 5857.5903320, -8690.4970703, 9546.0849609
4: -8160.4511719, 5299.2163086, -6584.4262695, 4295.7910156, -12451.3427734, 11871.0273438

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1862970, upper bound: 4551.3986492
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1828397, upper bound: 4552.1781399
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6465.6679688, 5197.9145508, -6465.6679688, 5197.9145508, -11659.2441406, 11659.2441406
1: -5188.6606445, 4998.1103516, -5188.6606445, 4998.1103516, -10185.4218750, 10185.4208984
2: -7431.6689453, 5431.6601562, -7431.6689453, 5431.6601562, -12847.5341797, 12847.5341797
3: -2865.7429199, 7345.2402344, -2865.7429199, 7345.2402344, -10186.5107422, 10186.5107422
4: -8250.3779297, 5354.0131836, -8250.3779297, 5354.0131836, -13591.3330078, 13591.3330078

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4629635, upper bound: 4552.1881294
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1828398, upper bound: 4552.1858269
time: 0.83 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.72 seconds
IS_A1_B1_B1_A1_B1, status: Status.VERIFIED, split count: 5, time: 4.72
Output dim: 0, lower bound: -4551.4690262, upper bound: 4551.4360878
IS_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -4551.4690262, upper bound: 4552.1987525
IS_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -4552.1892226, upper bound: 4551.4360878
IS_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -4552.1892226, upper bound: 4552.1987525
IS_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -4551.4360878, upper bound: 4552.1862971
IS_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -4552.1811339, upper bound: 4552.1828398
IS_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -4552.1846100, upper bound: 4551.4614286
IS_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -4552.1811339, upper bound: 4552.1820814
IS_A1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -4552.1896128, upper bound: 4551.1871985
IS_A1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -4552.1874217, upper bound: 4551.3949576
IS_A1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -4552.1878615, upper bound: 4552.1500239
IS_A1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -4552.1849632, upper bound: 4552.1746604
IS_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -4552.1862018, upper bound: 4551.9492265
IS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -4552.1855638, upper bound: 4552.1866626
IS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -4552.1862018, upper bound: 4551.9492265
IS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -4552.1855638, upper bound: 4552.1862412
IS_A2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -4551.1871985, upper bound: 4552.1896128
IS_A2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -4551.3949576, upper bound: 4552.1874217
IS_A2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -4552.1500239, upper bound: 4552.1880920
IS_A2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -4552.1746604, upper bound: 4552.1849632
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -4551.9492265, upper bound: 4552.1862018
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -4552.1866626, upper bound: 4552.1855638
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -4551.9492265, upper bound: 4552.1878615
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -4552.1866626, upper bound: 4552.1874041
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -4551.3949576, upper bound: 4552.1819804
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -4552.1746604, upper bound: 4552.1804615
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -4551.3949576, upper bound: 4552.1899945
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -4552.1746604, upper bound: 4552.1883855
IS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -4552.1862970, upper bound: 4551.3986492
IS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -4552.1828397, upper bound: 4552.1781399
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -4551.4629635, upper bound: 4552.1881294
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.72
Output dim: 0, lower bound: -4552.1828398, upper bound: 4552.1858269

## BFS IS instance: IS_A1_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -2864.9853516, 2335.5559082, -2854.9377441, 2325.1379395, -5190.1230469, 5190.4936523
1: -2295.3029785, 2250.7009277, -2286.3728027, 2239.5864258, -4534.8896484, 4537.0737305
2: -3284.9338379, 2445.8928223, -3270.2888184, 2435.0783691, -5720.0117188, 5716.1816406
3: -1288.4378662, 3259.7199707, -1282.8604736, 3243.4042969, -4531.8422852, 4542.5800781
4: -3655.1245117, 2403.8120117, -3640.6374512, 2394.5166016, -6049.6406250, 6044.4492188

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4427308, upper bound: 4552.2022458
time: 0.87 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4427308, upper bound: 4552.2022458
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -2890.1743164, 2353.1525879, -2829.3090820, 2307.1660156, -5197.3403320, 5182.4619141
1: -2314.7424316, 2266.5371094, -2266.5866699, 2223.4016113, -4538.1430664, 4533.1240234
2: -3311.0268555, 2464.3937988, -3243.7138672, 2416.2275391, -5727.2539062, 5708.1069336
3: -1298.1706543, 3283.7622070, -1272.9768066, 3218.7609863, -4516.9301758, 4556.7373047
4: -3685.6552734, 2423.2575684, -3609.5827637, 2374.6940918, -6060.3496094, 6032.8403320

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1876206, upper bound: 4551.4360878
time: 0.81 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1876206, upper bound: 4551.4360878
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -2890.1743164, 2353.1525879, -2854.9377441, 2325.1379395, -5215.3125000, 5208.0903320
1: -2314.7424316, 2266.5371094, -2286.3728027, 2239.5864258, -4554.3291016, 4552.9101562
2: -3311.0268555, 2464.3937988, -3270.2888184, 2435.0783691, -5746.1049805, 5734.6821289
3: -1298.1706543, 3283.7622070, -1282.8604736, 3243.4042969, -4541.5742188, 4566.6215820
4: -3685.6552734, 2423.2575684, -3640.6374512, 2394.5166016, -6080.1708984, 6063.8950195

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1876207, upper bound: 4552.1947892
time: 0.81 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1876207, upper bound: 4552.1947892
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -2829.3090820, 2307.1660156, -4141.2041016, 3330.3676758, -6159.6767578, 6448.3701172
1: -2266.5866699, 2223.4016113, -3329.8930664, 3205.2346191, -5471.8203125, 5553.2949219
2: -3243.7138672, 2416.2275391, -4775.1411133, 3475.1816406, -6718.8955078, 7191.3676758
3: -1272.9768066, 3218.7609863, -1829.5156250, 4716.5966797, -5989.5727539, 5048.2763672
4: -3609.5827637, 2374.6940918, -5292.0205078, 3421.3986816, -7030.9814453, 7666.7143555

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4360878, upper bound: 4551.4690260
time: 0.89 seconds

## Relational analysis of IS_A1_B1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4360878, upper bound: 4552.1892225
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -2854.9377441, 2325.1379395, -4160.4765625, 3345.7805176, -6200.7172852, 6485.6142578
1: -2286.3728027, 2239.5864258, -3345.3122559, 3219.9697266, -5506.3427734, 5584.8984375
2: -3270.2888184, 2435.0783691, -4797.0195312, 3491.4050293, -6761.6928711, 7232.0966797
3: -1282.8604736, 3243.4042969, -1838.1712646, 4737.9511719, -6020.8115234, 5081.5756836
4: -3640.6374512, 2394.5166016, -5316.2504883, 3437.5615234, -7078.1992188, 7710.7661133

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1987525, upper bound: 4551.4690260
time: 0.89 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1987525, upper bound: 4552.1892225
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4247.5839844, 3411.2246094, -4175.5634766, 3350.7539062, -7598.3378906, 7586.7880859
1: -3417.7436523, 3283.4750977, -3362.1945801, 3227.0385742, -6644.7822266, 6645.6684570
2: -4902.4863281, 3558.4304199, -4825.2534180, 3494.4333496, -8396.9199219, 8383.6835938
3: -1871.9594727, 4840.7377930, -1836.1400146, 4765.6860352, -6637.6450195, 6676.8779297
4: -5431.2021484, 3502.2399902, -5343.5141602, 3436.8771973, -8868.0771484, 8845.7529297

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4629633, upper bound: 4551.4614286
time: 1.02 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4629635, upper bound: 4551.4614286
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4267.7421875, 3427.3356934, -4186.9238281, 3361.6271973, -7629.3691406, 7614.2587891
1: -3433.8293457, 3298.9086914, -3369.0449219, 3235.5971680, -6669.4262695, 6667.9536133
2: -4925.3828125, 3575.4016113, -4832.7299805, 3506.0024414, -8431.3847656, 8408.1318359
3: -1881.0106201, 4863.0942383, -1843.7967529, 4772.0375977, -6653.0483398, 6706.8911133
4: -5456.5566406, 3519.1264648, -5354.3334961, 3450.2895508, -8906.8457031, 8873.4599609

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4629633, upper bound: 4552.1820814
time: 0.75 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4629633, upper bound: 4552.1820814
time: 2.83 seconds

## BFS IS instance: IS_A1_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -2814.1394043, 2288.9763184, -4479.0219727, 3615.1547852, -6429.2934570, 6767.9975586
1: -2254.3825684, 2205.0200195, -3592.1394043, 3480.6867676, -5735.0683594, 5797.1591797
2: -3225.5466309, 2397.2131348, -5141.6450195, 3782.5295410, -7008.0761719, 7538.8583984
3: -1262.6512451, 3197.5500488, -1996.2467041, 5091.6811523, -6354.3325195, 5193.7968750
4: -3588.6369629, 2356.9943848, -5721.3095703, 3725.9438477, -7314.5795898, 8078.3032227

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_B1_B1_A1

### Relational analysis result of IS_A1_B2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1855922, upper bound: 4551.1856565
time: 0.95 seconds

## Relational analysis of IS_A1_B2_B1_B1_B1_A2

### Relational analysis result of IS_A1_B2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1855922, upper bound: 4551.1862575
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -2950.6777344, 2401.2275391, -4971.9711914, 4020.7971191, -6971.4731445, 7373.1987305
1: -2363.7670898, 2313.0334473, -3985.2780762, 3870.9291992, -6234.6962891, 6298.3115234
2: -3381.3503418, 2515.3278809, -5702.8066406, 4206.5654297, -7587.9160156, 8218.1337891
3: -1325.3363037, 3351.5842285, -2218.2829590, 5650.0703125, -6975.4067383, 5569.8671875
4: -3762.7043457, 2473.1333008, -6347.5229492, 4143.5112305, -7906.2153320, 8820.6562500

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_B1_B2_A1

### Relational analysis result of IS_A1_B2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1831392, upper bound: 4551.3934990
time: 0.76 seconds

## Relational analysis of IS_A1_B2_B1_B1_B2_A2

### Relational analysis result of IS_A1_B2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1872107, upper bound: 4551.3941013
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -2834.0895996, 2304.6740723, -4511.9248047, 3639.9665527, -6474.0561523, 6816.5976562
1: -2270.4116211, 2220.1262207, -3618.2277832, 3503.8317871, -5774.2431641, 5838.3540039
2: -3248.3408203, 2413.7231445, -5177.4951172, 3808.8811035, -7057.2216797, 7591.2172852
3: -1271.3266602, 3219.7036133, -2010.2752686, 5125.5092773, -6396.8354492, 5229.9790039
4: -3613.8750000, 2373.3015137, -5762.2270508, 3752.5803223, -7366.4550781, 8135.5278320

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1839444, upper bound: 4552.1487650
time: 0.97 seconds

## Relational analysis of IS_A1_B2_B1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1879556, upper bound: 4552.1492340
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -2970.4340820, 2416.7614746, -5017.2783203, 4054.7910156, -7025.2246094, 7434.0390625
1: -2379.6340332, 2327.9750977, -4021.0563965, 3903.0412598, -6282.6748047, 6349.0312500
2: -3403.9311523, 2531.6525879, -5752.4824219, 4242.5092773, -7646.4404297, 8284.1347656
3: -1333.9206543, 3373.6022949, -2237.3461914, 5697.2138672, -7031.1333008, 5610.9482422
4: -3787.6931152, 2489.2673340, -6403.8979492, 4179.5209961, -7967.2138672, 8893.1640625

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4651408, upper bound: 4552.1746605
time: 0.88 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4651408, upper bound: 4552.1710640
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -2798.0192871, 2275.8630371, -5902.4619141, 4727.6738281, -7525.6928711, 8178.3251953
1: -2241.3710938, 2192.3952637, -4737.2656250, 4546.9023438, -6788.2734375, 6929.6611328
2: -3206.6320801, 2383.5732422, -6784.6796875, 4939.5903320, -8146.2221680, 9168.2529297
3: -1255.5870361, 3178.2351074, -2605.8508301, 6697.9580078, -7951.4907227, 5784.0859375
4: -3567.7829590, 2343.7578125, -7531.8115234, 4869.5131836, -8437.2958984, 9875.5664062

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2037965, upper bound: 4551.0266900
time: 0.92 seconds

## Relational analysis of IS_A1_B2_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2019278, upper bound: 4551.9517258
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -2935.8173828, 2389.1713867, -6289.3193359, 5061.2558594, -7997.0727539, 8678.4902344
1: -2351.7700195, 2301.4243164, -5045.8598633, 4866.4599609, -7218.2299805, 7347.2836914
2: -3363.9299316, 2502.7832031, -7224.7275391, 5289.9633789, -8653.8935547, 9727.5097656
3: -1318.8361816, 3333.9367676, -2791.0410156, 7144.1020508, -8462.9375000, 6124.9775391
4: -3743.4733887, 2460.9711914, -8023.8012695, 5214.2695312, -8957.7431641, 10484.7714844

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2031810, upper bound: 4551.4690260
time: 0.86 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2006910, upper bound: 4552.1891892
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4122.1201172, 3305.4670410, -5978.6865234, 4786.7617188, -8908.8808594, 9284.1533203
1: -3318.1379395, 3182.8261719, -4799.3916016, 4603.5590820, -7921.6967773, 7982.2167969
2: -4760.7182617, 3447.6413574, -6875.3281250, 5000.4335938, -9761.1523438, 10322.9697266
3: -1811.6857910, 4699.1625977, -2637.8525391, 6785.8408203, -8592.7939453, 7332.0605469
4: -5272.6923828, 3391.6967773, -7630.0805664, 4929.4306641, -10202.1230469, 11021.7744141

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1878690, upper bound: 4551.0197041
time: 0.78 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1860163, upper bound: 4551.9449425
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4240.1347656, 3406.3552246, -6363.1420898, 5118.8481445, -9358.9824219, 9769.4970703
1: -3411.5942383, 3278.4863281, -5105.9799805, 4921.5893555, -8333.1826172, 8384.4667969
2: -4893.2680664, 3553.4892578, -7312.7739258, 5349.3676758, -10242.6347656, 10866.2636719
3: -1869.3400879, 4831.7202148, -2822.4138184, 7229.1621094, -9098.2031250, 7649.0610352
4: -5420.9492188, 3497.4892578, -8119.0649414, 5272.7631836, -10693.7128906, 11616.5546875

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1855549, upper bound: 4551.4614286
time: 0.91 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1847888, upper bound: 4552.1820630
time: 1.63 seconds

## BFS IS instance: IS_A2_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -4479.0219727, 3615.1547852, -2805.3439941, 2282.1943359, -6761.2163086, 6420.4975586
1: -3592.1394043, 3480.6867676, -2247.2187500, 2198.4709473, -5790.6103516, 5727.9052734
2: -5141.6450195, 3782.5295410, -3215.2216797, 2390.0075684, -7531.6523438, 6997.7509766
3: -1996.2467041, 5091.6811523, -1258.7407227, 3187.7011719, -5183.9477539, 6350.4218750
4: -5721.3095703, 3725.9438477, -3577.2980957, 2349.9455566, -8071.2548828, 7303.2421875

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_A1_A1_B1

### Relational analysis result of IS_A2_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.1856565, upper bound: 4552.1855922
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_A1_A1_B2

### Relational analysis result of IS_A2_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.1862575, upper bound: 4552.1896128
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -4971.9711914, 4020.7971191, -2941.9443359, 2394.5336914, -7366.5048828, 6962.7397461
1: -3985.2780762, 3870.9291992, -2356.6608887, 2306.5563965, -6291.8344727, 6227.5898438
2: -5702.8066406, 4206.5654297, -3371.0993652, 2508.2197266, -8211.0263672, 7577.6645508
3: -2218.2829590, 5650.0703125, -1321.4779053, 3341.6904297, -5559.9731445, 6971.5473633
4: -6347.5229492, 4143.5112305, -3751.4633789, 2466.1750488, -8813.6982422, 7894.9746094

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_A1_A2_B1

### Relational analysis result of IS_A2_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.3934990, upper bound: 4552.1831392
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_A1_A2_B2

### Relational analysis result of IS_A2_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.3941013, upper bound: 4552.1872107
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -4511.9248047, 3639.9665527, -2825.2883301, 2297.8955078, -6809.8188477, 6465.2539062
1: -3618.2277832, 3503.8317871, -2263.2441406, 2213.5634766, -5831.7910156, 5767.0761719
2: -5177.4951172, 3808.8811035, -3238.0134277, 2406.5195312, -7584.0146484, 7046.8935547
3: -2010.2752686, 5125.5092773, -1267.4199219, 3209.8405762, -5220.1157227, 6392.9287109
4: -5762.2270508, 3752.5803223, -3602.5349121, 2366.2563477, -8128.4833984, 7355.1147461

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_A2_A1_B1

### Relational analysis result of IS_A2_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1487650, upper bound: 4552.1839444
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_A2_A1_B2

### Relational analysis result of IS_A2_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1492340, upper bound: 4552.1879556
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -5017.2783203, 4054.7910156, -2961.7177734, 2410.0881348, -7427.3662109, 7016.5087891
1: -4021.0563965, 3903.0412598, -2372.5410156, 2321.5178223, -6342.5742188, 6275.5820312
2: -5752.4824219, 4242.5092773, -3393.7001953, 2524.5693359, -8277.0517578, 7636.2089844
3: -2237.3461914, 5697.2138672, -1330.0771484, 3363.7304688, -5601.0751953, 7027.2910156
4: -6403.8979492, 4179.5209961, -3776.4738770, 2482.3317871, -8886.2294922, 7955.9946289

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_A2_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1746604, upper bound: 4551.4651408
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A1_A2_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1746605, upper bound: 4552.1815131
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5902.4589844, 4727.6713867, -2791.8894043, 2271.1403809, -8173.5991211, 7519.5600586
1: -4737.2631836, 4546.8999023, -2236.3779297, 2187.8239746, -6925.0869141, 6783.2778320
2: -6784.6762695, 4939.5878906, -3199.4355469, 2378.5554199, -9163.2314453, 8139.0229492
3: -2605.8496094, 6697.9550781, -1252.8651123, 3171.3417969, -5777.1914062, 7948.7690430
4: -7531.8076172, 4869.5112305, -3559.8850098, 2338.8459473, -9870.6523438, 8429.3945312

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.0266900, upper bound: 4552.2037965
time: 1.01 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9517258, upper bound: 4552.2019278
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6289.3193359, 5061.2558594, -2929.7441406, 2384.5187988, -8673.8378906, 7990.9995117
1: -5045.8598633, 4866.4599609, -2346.8269043, 2296.9226074, -7342.7822266, 7213.2871094
2: -7224.7275391, 5289.9633789, -3356.7971191, 2497.8454590, -9722.5722656, 8646.7607422
3: -2791.0410156, 7144.1020508, -1316.1584473, 3327.0532227, -6118.0937500, 8460.2587891
4: -8023.8012695, 5214.2695312, -3735.6567383, 2456.1367188, -10479.9375000, 8949.9257812

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4690260, upper bound: 4552.2031810
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9517258, upper bound: 4552.2006910
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5978.6865234, 4786.7617188, -4122.1201172, 3305.4670410, -9284.1533203, 8908.8818359
1: -4799.3916016, 4603.5590820, -3318.1379395, 3182.8261719, -7982.2172852, 7921.6962891
2: -6875.3281250, 5000.4335938, -4760.7182617, 3447.6413574, -10322.9697266, 9761.1523438
3: -2637.8525391, 6785.8408203, -1811.6857910, 4699.1625977, -7332.0605469, 8592.7949219
4: -7630.0805664, 4929.4306641, -5272.6923828, 3391.6967773, -11021.7744141, 10202.1230469

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.0197041, upper bound: 4552.1874312
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9449425, upper bound: 4552.1854982
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6363.1420898, 5118.8481445, -4240.1347656, 3406.3552246, -9769.4970703, 9358.9824219
1: -5105.9799805, 4921.5893555, -3411.5942383, 3278.4863281, -8384.4667969, 8333.1826172
2: -7312.7739258, 5349.3676758, -4893.2680664, 3553.4892578, -10866.2636719, 10242.6347656
3: -2822.4138184, 7229.1621094, -1869.3400879, 4831.7202148, -7649.0615234, 9098.2031250
4: -8119.0649414, 5272.7631836, -5420.9492188, 3497.4892578, -11616.5546875, 10693.7128906

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4629635, upper bound: 4552.1869852
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1828398, upper bound: 4552.1843941
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5030.1420898, 4068.0788574, -5141.1723633, 4153.9165039, -9184.0576172, 9209.2509766
1: -4032.0764160, 3916.5642090, -4121.1528320, 3998.8049316, -8030.8798828, 8037.7163086
2: -5769.9741211, 4256.2983398, -5896.1274414, 4346.6757812, -10111.4892578, 10147.5771484
3: -2244.9921875, 5716.9531250, -2292.9245605, 5838.2089844, -8081.0488281, 8009.8769531
4: -6422.0834961, 4192.4262695, -6562.5346680, 4281.7187500, -10696.2099609, 10747.9541016

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4004611, upper bound: 4551.4009710
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4004611, upper bound: 4552.1804615
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5074.3295898, 4101.2333984, -5158.3984375, 4167.5395508, -9241.8691406, 9259.6318359
1: -4067.0036621, 3947.8349609, -4135.0488281, 4011.9240723, -8078.9272461, 8082.8828125
2: -5818.3769531, 4291.3007812, -5915.9174805, 4360.9184570, -10178.1035156, 10202.6074219
3: -2263.4909668, 5762.7929688, -2300.4028320, 5857.5903320, -8118.8564453, 8063.1958008
4: -6477.0117188, 4227.5278320, -6584.4262695, 4295.7910156, -10769.0546875, 10805.0605469

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_A2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4533.7228740, upper bound: 4545.2242302
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1800169, upper bound: 4552.1804232
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5030.1420898, 4068.0788574, -6374.2836914, 5126.7500000, -10150.8261719, 10442.3623047
1: -4032.0764160, 3916.5642090, -5114.4282227, 4929.9648438, -8958.8603516, 9030.9912109
2: -5769.9741211, 4256.2983398, -7323.8188477, 5358.0786133, -11116.3125000, 11570.4746094
3: -2244.9921875, 5716.9531250, -2826.9489746, 7240.9111328, -9464.3505859, 8540.4755859
4: -6422.0834961, 4192.4262695, -8133.0546875, 5281.4775391, -11688.4394531, 12318.6679688

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.3949576, upper bound: 4551.4682767
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.3949576, upper bound: 4552.1883854
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5074.3295898, 4101.2333984, -6393.0556641, 5141.6889648, -10213.5312500, 10494.2597656
1: -4067.0036621, 3947.8349609, -5129.5156250, 4944.3125000, -9010.8916016, 9077.3496094
2: -5818.3769531, 4291.3007812, -7345.2299805, 5373.7558594, -11184.3554688, 11627.0957031
3: -2263.4909668, 5762.7929688, -2835.1689453, 7261.8808594, -9503.7470703, 8594.9609375
4: -6477.0117188, 4227.5278320, -8156.7954102, 5296.9667969, -11762.6806641, 12377.6093750

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1746605, upper bound: 4551.4682767
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1746605, upper bound: 4552.1883854
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -6374.2836914, 5126.7500000, -5030.1420898, 4068.0788574, -10442.3623047, 10150.8271484
1: -5114.4282227, 4929.9648438, -4032.0764160, 3916.5642090, -9030.9912109, 8958.8603516
2: -7323.8188477, 5358.0786133, -5769.9741211, 4256.2983398, -11570.4746094, 11116.3125000
3: -2826.9489746, 7240.9111328, -2244.9921875, 5716.9531250, -8540.4765625, 9464.3496094
4: -8133.0546875, 5281.4775391, -6422.0834961, 4192.4262695, -12318.6669922, 11688.4394531

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4676908, upper bound: 4551.3986492
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4676908, upper bound: 4551.3986492
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -6393.0556641, 5141.6889648, -5074.3295898, 4101.2333984, -10494.2597656, 10213.5312500
1: -5129.5161133, 4944.3125000, -4067.0036621, 3947.8349609, -9077.3505859, 9010.8916016
2: -7345.2299805, 5373.7558594, -5818.3769531, 4291.3007812, -11627.0957031, 11184.3554688
3: -2835.1687012, 7261.8808594, -2263.4909668, 5762.7929688, -8594.9609375, 9503.7470703
4: -8156.7949219, 5296.9663086, -6477.0117188, 4227.5278320, -12377.6093750, 11762.6806641

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4676908, upper bound: 4552.1781399
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4676908, upper bound: 4552.1781399
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6366.1660156, 5119.2890625, -6446.3999023, 5182.5800781, -11542.4501953, 11557.8535156
1: -5110.7729492, 4923.2451172, -5173.2172852, 4983.3857422, -10090.4931641, 10091.8007812
2: -7322.4482422, 5348.0849609, -7409.6840820, 5415.5717773, -12718.9951172, 12737.4667969
3: -2820.9536133, 7239.8808594, -2857.3107910, 7323.7172852, -10116.5966797, 10072.1220703
4: -8126.7597656, 5270.9204102, -8226.0058594, 5338.1235352, -13449.1435547, 13479.5908203

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4629633, upper bound: 4551.4655518
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4629633, upper bound: 4552.1858269
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6390.3413086, 5137.2963867, -6465.6679688, 5197.9145508, -11584.8183594, 11595.4892578
1: -5128.1879883, 4939.5961914, -5188.6606445, 4998.1103516, -10125.4355469, 10124.1347656
2: -7344.9951172, 5367.6391602, -7431.6689453, 5431.6601562, -12761.6904297, 12779.8447266
3: -2831.8232422, 7260.3583984, -2865.7429199, 7345.2402344, -10149.3730469, 10101.5341797
4: -8154.4711914, 5291.2568359, -8250.3779297, 5354.0131836, -13496.5107422, 13524.7314453

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1828397, upper bound: 4551.4655518
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1828397, upper bound: 4552.1858269
time: 1.00 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.50 seconds
IS_A1_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4551.4427308, upper bound: 4552.2022458
IS_A1_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4551.4427308, upper bound: 4552.2022458
IS_A1_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4552.1876206, upper bound: 4551.4360878
IS_A1_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4552.1876206, upper bound: 4551.4360878
IS_A1_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4552.1876207, upper bound: 4552.1947892
IS_A1_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4552.1876207, upper bound: 4552.1947892
IS_A1_B1_B2_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.50
Output dim: 0, lower bound: -4551.4360878, upper bound: 4551.4690260
IS_A1_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4551.4360878, upper bound: 4552.1892225
IS_A1_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4552.1987525, upper bound: 4551.4690260
IS_A1_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4552.1987525, upper bound: 4552.1892225
IS_A1_B1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 4.50
Output dim: 0, lower bound: -4551.4629633, upper bound: 4551.4614286
IS_A1_B1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 4.50
Output dim: 0, lower bound: -4551.4629635, upper bound: 4551.4614286
IS_A1_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4551.4629633, upper bound: 4552.1820814
IS_A1_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4551.4629633, upper bound: 4552.1820814
IS_A1_B2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4552.1855922, upper bound: 4551.1856565
IS_A1_B2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4552.1855922, upper bound: 4551.1862575
IS_A1_B2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4552.1831392, upper bound: 4551.3934990
IS_A1_B2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4552.1872107, upper bound: 4551.3941013
IS_A1_B2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4552.1839444, upper bound: 4552.1487650
IS_A1_B2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4552.1879556, upper bound: 4552.1492340
IS_A1_B2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4551.4651408, upper bound: 4552.1746605
IS_A1_B2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4551.4651408, upper bound: 4552.1710640
IS_A1_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4552.2037965, upper bound: 4551.0266900
IS_A1_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4552.2019278, upper bound: 4551.9517258
IS_A1_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4552.2031810, upper bound: 4551.4690260
IS_A1_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4552.2006910, upper bound: 4552.1891892
IS_A1_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4552.1878690, upper bound: 4551.0197041
IS_A1_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4552.1860163, upper bound: 4551.9449425
IS_A1_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4552.1855549, upper bound: 4551.4614286
IS_A1_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4552.1847888, upper bound: 4552.1820630
IS_A2_B1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4551.1856565, upper bound: 4552.1855922
IS_A2_B1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4551.1862575, upper bound: 4552.1896128
IS_A2_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4551.3934990, upper bound: 4552.1831392
IS_A2_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4551.3941013, upper bound: 4552.1872107
IS_A2_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4552.1487650, upper bound: 4552.1839444
IS_A2_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4552.1492340, upper bound: 4552.1879556
IS_A2_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4552.1746604, upper bound: 4551.4651408
IS_A2_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4552.1746605, upper bound: 4552.1815131
IS_A2_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4551.0266900, upper bound: 4552.2037965
IS_A2_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4551.9517258, upper bound: 4552.2019278
IS_A2_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4551.4690260, upper bound: 4552.2031810
IS_A2_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4551.9517258, upper bound: 4552.2006910
IS_A2_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4551.0197041, upper bound: 4552.1874312
IS_A2_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4551.9449425, upper bound: 4552.1854982
IS_A2_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4551.4629635, upper bound: 4552.1869852
IS_A2_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4552.1828398, upper bound: 4552.1843941
IS_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.50
Output dim: 0, lower bound: -4551.4004611, upper bound: 4551.4009710
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4551.4004611, upper bound: 4552.1804615
IS_A2_B2_A1_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 4.50
Output dim: 0, lower bound: -4533.7228740, upper bound: 4545.2242302
IS_A2_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4552.1800169, upper bound: 4552.1804232
IS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.50
Output dim: 0, lower bound: -4551.3949576, upper bound: 4551.4682767
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4551.3949576, upper bound: 4552.1883854
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4552.1746605, upper bound: 4551.4682767
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4552.1746605, upper bound: 4552.1883854
IS_A2_B2_A2_B1_B1_A1, status: Status.VERIFIED, split count: 6, time: 4.50
Output dim: 0, lower bound: -4551.4676908, upper bound: 4551.3986492
IS_A2_B2_A2_B1_B1_A2, status: Status.VERIFIED, split count: 6, time: 4.50
Output dim: 0, lower bound: -4551.4676908, upper bound: 4551.3986492
IS_A2_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4551.4676908, upper bound: 4552.1781399
IS_A2_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4551.4676908, upper bound: 4552.1781399
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.50
Output dim: 0, lower bound: -4551.4629633, upper bound: 4551.4655518
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4551.4629633, upper bound: 4552.1858269
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4552.1828397, upper bound: 4551.4655518
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 0, lower bound: -4552.1828397, upper bound: 4552.1858269

## BFS IS instance: IS_A1_B1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -2829.3090820, 2307.1660156, -2854.9377441, 2325.1379395, -5154.4472656, 5162.1035156
1: -2266.5866699, 2223.4016113, -2286.3728027, 2239.5864258, -4506.1728516, 4509.7744141
2: -3243.7138672, 2416.2275391, -3270.2888184, 2435.0783691, -5678.7915039, 5686.5151367
3: -1272.9768066, 3218.7609863, -1282.8604736, 3243.4042969, -4516.3803711, 4501.6215820
4: -3609.5827637, 2374.6940918, -3640.6374512, 2394.5166016, -6004.0976562, 6015.3315430

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4423639, upper bound: 4552.1715867
time: 0.83 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4427229, upper bound: 4552.2022210
time: 1.48 seconds

## BFS IS instance: IS_A1_B1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4054.9409180, 3259.4003906, -2854.9377441, 2325.1379395, -6380.0781250, 6114.3378906
1: -3262.9404297, 3138.5986328, -2286.3728027, 2239.5864258, -5502.5268555, 5424.9716797
2: -4681.6396484, 3400.1430664, -3270.2888184, 2435.0783691, -7116.7167969, 6670.4316406
3: -1788.1987305, 4625.9775391, -1282.8604736, 3243.4042969, -5031.6030273, 5908.8374023
4: -5186.1127930, 3345.4672852, -3640.6374512, 2394.5166016, -7580.6279297, 6986.1044922

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_A1

### Relational analysis result of IS_A1_B1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4411299, upper bound: 4552.2017677
time: 1.00 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_A2

### Relational analysis result of IS_A1_B1_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4412370, upper bound: 4552.2021667
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -2854.9377441, 2325.1379395, -2829.3090820, 2307.1660156, -5162.1035156, 5154.4472656
1: -2286.3728027, 2239.5864258, -2266.5866699, 2223.4016113, -4509.7744141, 4506.1728516
2: -3270.2888184, 2435.0783691, -3243.7138672, 2416.2275391, -5686.5151367, 5678.7915039
3: -1282.8604736, 3243.4042969, -1272.9768066, 3218.7609863, -4501.6210938, 4516.3808594
4: -3640.6374512, 2394.5166016, -3609.5827637, 2374.6940918, -6015.3315430, 6004.0981445

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_A2_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1863371, upper bound: 4551.4346747
time: 0.94 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1887721, upper bound: 4551.4352770
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -4079.9606934, 3280.4335938, -2829.3090820, 2307.1660156, -6387.1269531, 6109.7426758
1: -3280.5886230, 3156.9819336, -2266.5866699, 2223.4016113, -5503.9902344, 5423.5678711
2: -4704.5625000, 3422.3588867, -3243.7138672, 2416.2275391, -7120.7890625, 6666.0727539
3: -1801.2983398, 4647.1855469, -1272.9768066, 3218.7609863, -5020.0590820, 5920.1616211
4: -5214.2836914, 3369.2727051, -3609.5827637, 2374.6940918, -7588.9775391, 6978.8540039

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_A2_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1863371, upper bound: 4551.4346747
time: 0.81 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1887721, upper bound: 4551.4352770
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -2854.9377441, 2325.1379395, -2854.9377441, 2325.1379395, -5180.0756836, 5180.0756836
1: -2286.3728027, 2239.5864258, -2286.3728027, 2239.5864258, -4525.9589844, 4525.9589844
2: -3270.2888184, 2435.0783691, -3270.2888184, 2435.0783691, -5705.3666992, 5705.3666992
3: -1282.8604736, 3243.4042969, -1282.8604736, 3243.4042969, -4526.2646484, 4526.2646484
4: -3640.6374512, 2394.5166016, -3640.6374512, 2394.5166016, -6035.1542969, 6035.1538086

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1871938, upper bound: 4552.1942116
time: 0.78 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1863449, upper bound: 4552.1942116
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4082.2253418, 3282.1577148, -2854.9377441, 2325.1379395, -6407.3618164, 6137.0957031
1: -3282.4580078, 3158.6499023, -2286.3728027, 2239.5864258, -5522.0444336, 5445.0224609
2: -4707.2709961, 3424.1333008, -3270.2888184, 2435.0783691, -7142.3486328, 6694.4218750
3: -1802.2034912, 4649.8281250, -1282.8604736, 3243.4042969, -5045.6079102, 5932.6884766
4: -5217.2456055, 3370.9956055, -3640.6374512, 2394.5166016, -7611.7617188, 7011.6328125

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1838736, upper bound: 4552.1936345
time: 0.99 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_A2_A2

### Relational analysis result of IS_A1_B1_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1863449, upper bound: 4552.1942116
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -2829.3090820, 2307.1660156, -4079.9606934, 3280.4335938, -6109.7426758, 6387.1269531
1: -2266.5866699, 2223.4016113, -3280.5886230, 3156.9819336, -5423.5678711, 5503.9902344
2: -3243.7138672, 2416.2275391, -4704.5625000, 3422.3588867, -6666.0727539, 7120.7895508
3: -1272.9768066, 3218.7609863, -1801.2983398, 4647.1855469, -5920.1621094, 5020.0590820
4: -3609.5827637, 2374.6940918, -5214.2836914, 3369.2727051, -6978.8540039, 7588.9775391

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B2_A1_A1_B2_B1

### Relational analysis result of IS_A1_B1_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4346747, upper bound: 4552.1863371
time: 0.81 seconds

## Relational analysis of IS_A1_B1_B2_A1_A1_B2_B2

### Relational analysis result of IS_A1_B1_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4352770, upper bound: 4552.1903447
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -2854.9377441, 2325.1379395, -4069.7946777, 3271.0456543, -6125.9833984, 6394.9316406
1: -2286.3728027, 2239.5864258, -3274.7546387, 3149.8955078, -5436.2685547, 5514.3408203
2: -3270.2888184, 2435.0783691, -4698.3183594, 3412.4604492, -6682.7490234, 7133.3964844
3: -1282.8604736, 3243.4042969, -1794.6108398, 4642.2182617, -5925.0781250, 5038.0151367
4: -3640.6374512, 2394.5166016, -5204.7719727, 3357.6091309, -6998.2465820, 7599.2875977

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B2_A1_A2_B1_B1

### Relational analysis result of IS_A1_B1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1975991, upper bound: 4551.4669590
time: 0.86 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2_B1_B2

### Relational analysis result of IS_A1_B1_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1980786, upper bound: 4551.4676057
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -2854.9377441, 2325.1379395, -4082.2253418, 3282.1577148, -6137.0957031, 6407.3618164
1: -2286.3728027, 2239.5864258, -3282.4580078, 3158.6499023, -5445.0224609, 5522.0444336
2: -3270.2888184, 2435.0783691, -4707.2709961, 3424.1333008, -6694.4218750, 7142.3486328
3: -1282.8604736, 3243.4042969, -1802.2034912, 4649.8281250, -5932.6884766, 5045.6074219
4: -3640.6374512, 2394.5166016, -5217.2456055, 3370.9956055, -7011.6328125, 7611.7617188

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B2_A1_A2_B2_B1

### Relational analysis result of IS_A1_B1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1975991, upper bound: 4552.1809593
time: 0.98 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2_B2_B2

### Relational analysis result of IS_A1_B1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1980786, upper bound: 4552.1849811
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4175.5634766, 3350.7539062, -4186.9238281, 3361.6271973, -7537.1904297, 7537.6777344
1: -3362.1945801, 3227.0385742, -3369.0449219, 3235.5971680, -6597.7915039, 6596.0834961
2: -4825.2534180, 3494.4333496, -4832.7299805, 3506.0024414, -8331.2558594, 8327.1630859
3: -1836.1400146, 4765.6860352, -1843.7967529, 4772.0375977, -6608.1777344, 6609.4819336
4: -5343.5141602, 3436.8771973, -5354.3334961, 3450.2895508, -8793.8007812, 8791.2109375

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_A1

### Relational analysis result of IS_A1_B1_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4616291, upper bound: 4552.1811328
time: 0.82 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_A2

### Relational analysis result of IS_A1_B1_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4622762, upper bound: 4552.1813725
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4186.9238281, 3361.6271973, -4186.9238281, 3361.6271973, -7548.5507812, 7548.5507812
1: -3369.0449219, 3235.5971680, -3369.0449219, 3235.5971680, -6604.6420898, 6604.6420898
2: -4832.7299805, 3506.0024414, -4832.7299805, 3506.0024414, -8338.7324219, 8338.7324219
3: -1843.7967529, 4772.0375977, -1843.7967529, 4772.0375977, -6615.8344727, 6615.8344727
4: -5354.3334961, 3450.2895508, -5354.3334961, 3450.2895508, -8804.6230469, 8804.6230469

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B2_A2_B2_A2_A1

### Relational analysis result of IS_A1_B1_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4616291, upper bound: 4552.1775144
time: 0.89 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_A2_A2

### Relational analysis result of IS_A1_B1_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4622764, upper bound: 4552.1778853
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -2727.5856934, 2220.1430664, -4466.0517578, 3604.7172852, -6332.3022461, 6686.1938477
1: -2184.7106934, 2138.5202637, -3581.7104492, 3470.6027832, -5655.3134766, 5720.2304688
2: -3125.7358398, 2325.1081543, -5126.7275391, 3771.5910645, -6897.3271484, 7451.8359375
3: -1225.0621338, 3097.9785156, -1990.5036621, 5076.9331055, -6301.9951172, 5088.4824219
4: -3478.6054688, 2286.2790527, -5704.7910156, 3715.2250977, -7193.8295898, 7991.0693359

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4690279, upper bound: 4551.1856565
time: 0.84 seconds

## Relational analysis of IS_A1_B2_B1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_B1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4690279, upper bound: 4551.1856565
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -3533.2097168, 2856.1003418, -4463.5297852, 3603.2023926, -7136.4121094, 7319.6298828
1: -2829.8522949, 2752.9653320, -3579.7319336, 3469.0834961, -6298.9350586, 6332.6967773
2: -4046.9387207, 2996.7109375, -5123.9458008, 3769.9824219, -7816.9208984, 8120.6557617
3: -1584.4357910, 4000.6169434, -1989.7061768, 5074.2866211, -6658.7226562, 5990.3222656
4: -4502.2832031, 2943.7612305, -5701.6718750, 3713.7077637, -8215.9912109, 8645.4306641

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4696777, upper bound: 4551.1862575
time: 0.81 seconds

## Relational analysis of IS_A1_B2_B1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_B1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4696777, upper bound: 4551.1862575
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -2865.3249512, 2333.2890625, -4957.7744141, 4009.5021973, -6874.8271484, 7291.0625000
1: -2295.0725098, 2247.3449707, -3973.8315430, 3860.0112305, -6155.0825195, 6221.1752930
2: -3282.9899902, 2444.1762695, -5686.4350586, 4194.7480469, -7477.7382812, 8130.6108398
3: -1288.3360596, 3253.7534180, -2212.1101074, 5633.9321289, -6922.2680664, 5465.8627930
4: -3654.1110840, 2403.3571777, -6329.4228516, 4131.9370117, -7786.0478516, 8732.7792969

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B2_B1_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4667779, upper bound: 4551.3934990
time: 0.80 seconds

## Relational analysis of IS_A1_B2_B1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B2_B1_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4667779, upper bound: 4551.3934990
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -3677.8664551, 2975.5410156, -4958.0869141, 4009.8801270, -7687.7465820, 7933.6274414
1: -2945.7907715, 2867.7521973, -3974.1328125, 3860.3640137, -6806.1547852, 6841.8837891
2: -4212.5146484, 3122.0114746, -5686.8920898, 4195.1142578, -8407.6269531, 8808.9023438
3: -1650.6917725, 4165.9389648, -2212.2473145, 5634.4501953, -7285.1420898, 6378.1860352
4: -4686.8715820, 3067.2263184, -6329.8496094, 4132.2832031, -8819.1542969, 9397.0761719

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B1_B1_B2_A2_A1

### Relational analysis result of IS_A1_B2_B1_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4674262, upper bound: 4551.3941013
time: 0.81 seconds

## Relational analysis of IS_A1_B2_B1_B1_B2_A2_A2

### Relational analysis result of IS_A1_B2_B1_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4674262, upper bound: 4551.3941013
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -2747.4033203, 2235.7185059, -4499.4780273, 3629.9216309, -6377.3232422, 6735.1962891
1: -2200.6228027, 2153.5065918, -3608.2048340, 3494.1298828, -5694.7529297, 5761.7114258
2: -3148.3771973, 2341.4863281, -5163.1606445, 3798.3420410, -6946.7187500, 7504.6469727
3: -1233.6849365, 3120.0070801, -2004.7333984, 5111.3437500, -6345.0288086, 5124.7397461
4: -3503.6723633, 2302.4689941, -5746.3500977, 3742.2519531, -7245.9243164, 8048.8188477

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4672619, upper bound: 4552.1487651
time: 0.87 seconds

## Relational analysis of IS_A1_B2_B1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4672619, upper bound: 4552.1445753
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -3552.3559570, 2871.1547852, -4496.8154297, 3628.3107910, -7180.6669922, 7367.9697266
1: -2845.2436523, 2767.4475098, -3606.1074219, 3492.5156250, -6337.7573242, 6373.5546875
2: -4068.8420410, 3012.4682617, -5160.2041016, 3796.6464844, -7865.4882812, 8172.6723633
3: -1592.7176514, 4022.0173340, -2003.8804932, 5108.5488281, -6701.2666016, 6025.8974609
4: -4526.5390625, 2959.3176270, -5743.0405273, 3740.6525879, -8267.1904297, 8702.3583984

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4679111, upper bound: 4552.1492340
time: 0.85 seconds

## Relational analysis of IS_A1_B2_B1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4679111, upper bound: 4552.1450523
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -2846.8686523, 2320.9636230, -5016.2348633, 4053.9638672, -6900.8310547, 7337.1982422
1: -2280.6723633, 2236.6364746, -4020.2236328, 3902.2438965, -6182.9160156, 6256.8598633
2: -3263.9316406, 2430.6298828, -5751.2919922, 4241.6455078, -7505.5771484, 8181.9218750
3: -1280.3332520, 3238.7390137, -2236.8952637, 5696.0346680, -6976.3681641, 5475.6342773
4: -3631.8615723, 2388.7741699, -6402.5732422, 4178.6743164, -7810.5356445, 8791.3476562

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4638668, upper bound: 4552.1734676
time: 0.94 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4646309, upper bound: 4552.1739417
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -2872.3154297, 2338.7680664, -5016.6523438, 4054.2939453, -6926.6093750, 7355.4199219
1: -2300.3159180, 2252.6894531, -4020.5563965, 3902.5625000, -6202.8779297, 6273.2456055
2: -3290.3271484, 2449.3542480, -5751.7680664, 4241.9912109, -7532.3173828, 8201.1220703
3: -1290.1838379, 3263.0998535, -2237.0751953, 5696.5053711, -6986.6884766, 5500.1748047
4: -3662.7233887, 2408.4370117, -6403.1025391, 4179.0126953, -7841.7363281, 8811.5390625

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4639831, upper bound: 4552.1698362
time: 0.81 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4646309, upper bound: 4552.1704937
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -2777.9104004, 2260.0366211, -5809.3647461, 4654.6210938, -7432.5312500, 8069.4013672
1: -2225.2097168, 2177.1701660, -4663.9765625, 4477.2641602, -6702.4736328, 6841.1459961
2: -3183.6447754, 2366.9287109, -6682.3139648, 4862.2055664, -8045.8505859, 9049.2421875
3: -1246.8428955, 3155.8457031, -2564.4072266, 6598.9931641, -7843.2495117, 5720.2509766
4: -3542.3347168, 2327.3146973, -7416.0366211, 4792.4155273, -8334.7500000, 9743.3476562

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B2_B2_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4395840, upper bound: 4551.0266900
time: 1.19 seconds

## Relational analysis of IS_A1_B2_B2_A1_B1_B1_A2

### Relational analysis result of IS_A1_B2_B2_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4395840, upper bound: 4551.0266900
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -2798.0192871, 2275.8630371, -5827.4008789, 4667.1557617, -7465.1748047, 8103.2636719
1: -2241.3710938, 2192.3952637, -4676.8955078, 4488.5107422, -6729.8818359, 6869.2905273
2: -3206.6320801, 2383.5732422, -6698.2749023, 4875.6279297, -8082.2597656, 9081.8476562
3: -1255.5870361, 3178.2351074, -2571.8713379, 6613.3979492, -7866.8095703, 5750.1059570
4: -3567.7829590, 2343.7578125, -7436.2890625, 4806.8637695, -8374.6464844, 9780.0468750

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4395840, upper bound: 4551.9517258
time: 0.86 seconds

## Relational analysis of IS_A1_B2_B2_A1_B1_B2_A2

### Relational analysis result of IS_A1_B2_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4395840, upper bound: 4551.9517258
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -2915.9121094, 2373.4946289, -6188.6411133, 4982.3999023, -7898.3120117, 8562.1357422
1: -2335.7822266, 2286.3488770, -4966.3251953, 4791.3320312, -7127.1137695, 7252.6733398
2: -3341.1687012, 2486.3098145, -7113.8642578, 5206.3671875, -8547.5361328, 9600.1738281
3: -1310.1826172, 3311.7241211, -2746.3737793, 7037.3876953, -8347.5703125, 6058.0976562
4: -3718.2893066, 2444.6938477, -7898.4248047, 5131.1240234, -8849.4130859, 10343.1171875

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4381556, upper bound: 4551.4690260
time: 0.76 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4381556, upper bound: 4551.4690260
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -2935.8173828, 2389.1713867, -6213.4360352, 5000.4941406, -7936.3110352, 8602.6074219
1: -2351.7700195, 2301.4243164, -4984.6796875, 4807.8569336, -7159.6269531, 7286.1040039
2: -3363.9299316, 2502.7832031, -7137.2919922, 5225.8642578, -8589.7939453, 9640.0742188
3: -1318.8361816, 3333.9367676, -2757.0690918, 7058.6982422, -8377.5341797, 6091.0058594
4: -3743.4733887, 2460.9711914, -7927.1357422, 5151.4047852, -8894.8759766, 10388.1054688

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B2_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4381556, upper bound: 4552.1891893
time: 0.81 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4381556, upper bound: 4552.1891893
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -4101.8530273, 3289.2414551, -5887.9448242, 4715.3886719, -8817.2421875, 9177.1865234
1: -3301.9875488, 3167.2954102, -4728.2705078, 4535.6669922, -7837.6542969, 7895.5654297
2: -4737.7377930, 3430.5485840, -6776.0810547, 4924.7973633, -9658.8886719, 10206.6289062
3: -1802.5330811, 4676.7163086, -2597.3303223, 6689.7460938, -8487.0371094, 7265.1728516
4: -5247.2387695, 3374.6169434, -7517.7158203, 4854.0903320, -10101.3281250, 10892.3330078

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_B2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B2_B2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4663238, upper bound: 4551.0197041
time: 0.80 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B2_B2_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4663238, upper bound: 4551.0197041
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -4122.1201172, 3305.4670410, -5904.6293945, 4726.8266602, -8848.9462891, 9210.0947266
1: -3318.1379395, 3182.8261719, -4739.8671875, 4545.7578125, -7863.8950195, 7922.6928711
2: -4760.7182617, 3447.6413574, -6790.2216797, 4937.0629883, -9695.2861328, 10237.8623047
3: -1811.6857910, 4699.1625977, -2604.1918945, 6702.3901367, -8509.1777344, 7295.0800781
4: -5272.6923828, 3391.6967773, -7535.9570312, 4867.3632812, -10140.0556641, 10927.6533203

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1819314, upper bound: 4551.9435213
time: 0.78 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1859970, upper bound: 4551.9440902
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -4220.0087891, 3390.2744141, -6263.6738281, 5040.8706055, -9260.8779297, 9653.9482422
1: -3395.5332031, 3263.0842285, -5028.1689453, 4847.3383789, -8242.8710938, 8291.2529297
2: -4870.4057617, 3536.5507812, -7203.6943359, 5266.5117188, -10136.9160156, 10740.2441406
3: -1860.3089600, 4809.4052734, -2778.0949707, 7124.1586914, -8983.4267578, 7578.8378906
4: -5395.6381836, 3480.6333008, -7995.6171875, 5190.4067383, -10586.0439453, 11476.2490234

Time for backsubstitution: 1.71 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=5687.5751953125
rel_dist={0: [-4552.283859528013, 4552.283859528014]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2683765, upper bound: 4552.2788348
time: 0.83 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2684405, upper bound: 4552.2684405
time: 0.89 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.87 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 1.87
Output dim: 0, lower bound: -4552.2683765, upper bound: 4552.2788348
IS_B2, status: Status.UNKNOWN, split count: 1, time: 1.87
Output dim: 0, lower bound: -4552.2684405, upper bound: 4552.2684405

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -3091.6228027, 2514.4470215, -2987.9899902, 2430.9243164, -5522.5468750, 5502.4370117
1: -2477.0888672, 2422.2722168, -2393.8061523, 2341.6025391, -4818.6914062, 4816.0776367
2: -3543.5996094, 2633.9428711, -3424.2626953, 2546.4648438, -6090.0644531, 6058.2055664
3: -1387.4975586, 3512.5122070, -1341.7878418, 3393.8977051, -4781.3955078, 4854.2998047
4: -3942.0732422, 2589.6149902, -3810.2302246, 2503.8750000, -6445.9482422, 6399.8442383

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2680127, upper bound: 4552.2680127
time: 0.93 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2680127, upper bound: 4552.2684405
time: 0.89 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -3064.8989258, 2494.3271484, -5188.6577148, 4191.8315430, -7256.7304688, 7682.9848633
1: -2455.5947266, 2403.0100098, -4159.4165039, 4035.3022461, -6490.8969727, 6562.4267578
2: -3512.8261719, 2613.0234375, -5950.7929688, 4386.3300781, -7899.1557617, 8563.8144531
3: -1376.5388184, 3483.1633301, -2313.8149414, 5892.1474609, -7268.6860352, 5796.9785156
4: -3908.0346680, 2568.8513184, -6623.0805664, 4320.7236328, -8228.7578125, 9191.9316406

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2684405, upper bound: 4552.2680127
time: 0.82 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2684405, upper bound: 4552.2684405
time: 0.85 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.29 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 3.29
Output dim: 0, lower bound: -4552.2680127, upper bound: 4552.2680127
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 3.29
Output dim: 0, lower bound: -4552.2680127, upper bound: 4552.2684405
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 3.29
Output dim: 0, lower bound: -4552.2684405, upper bound: 4552.2680127
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 3.29
Output dim: 0, lower bound: -4552.2684405, upper bound: 4552.2684405

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -2987.9899902, 2430.9243164, -2987.9899902, 2430.9243164, -5418.9140625, 5418.9140625
1: -2393.8061523, 2341.6025391, -2393.8061523, 2341.6025391, -4735.4086914, 4735.4086914
2: -3424.2626953, 2546.4648438, -3424.2626953, 2546.4648438, -5970.7275391, 5970.7275391
3: -1341.7878418, 3393.8977051, -1341.7878418, 3393.8977051, -4735.6855469, 4735.6855469
4: -3810.2302246, 2503.8750000, -3810.2302246, 2503.8750000, -6314.1054688, 6314.1054688

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1869657, upper bound: 4552.2051173
time: 0.77 seconds

## Relational analysis of IS_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844629, upper bound: 4552.1844628
time: 0.82 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -5181.7377930, 4186.3496094, -2987.9899902, 2430.9243164, -7612.6621094, 7174.3393555
1: -4153.8891602, 4030.0212402, -2393.8061523, 2341.6025391, -6495.4916992, 6423.8271484
2: -5942.8955078, 4380.5971680, -3424.2626953, 2546.4648438, -8489.3583984, 7804.8598633
3: -2310.8229980, 5884.3271484, -1341.7878418, 3393.8977051, -5704.7207031, 7226.1152344
4: -6614.2963867, 4315.1059570, -3810.2302246, 2503.8750000, -9118.1718750, 8125.3354492

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1870487, upper bound: 4552.1869816
time: 0.80 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844628, upper bound: 4552.1868456
time: 0.92 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -2979.2597656, 2424.2385254, -5181.7377930, 4186.3496094, -7165.6093750, 7605.9755859
1: -2386.7011719, 2335.1318359, -4153.8881836, 4030.0212402, -6416.7226562, 6489.0200195
2: -3414.0134277, 2539.3676758, -5942.8955078, 4380.5976562, -7794.6113281, 8482.2636719
3: -1337.9373779, 3384.0065918, -2310.8229980, 5884.3271484, -7222.2646484, 5694.8295898
4: -3798.9916992, 2496.9252930, -6614.2958984, 4315.1054688, -8114.0971680, 9111.2207031

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1865816, upper bound: 4552.1870487
time: 1.09 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844629, upper bound: 4552.1844628
time: 0.83 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -5188.6577148, 4191.8315430, -5188.6577148, 4191.8315430, -9380.4892578, 9380.4892578
1: -4159.4165039, 4035.3022461, -4159.4165039, 4035.3022461, -8194.7187500, 8194.7187500
2: -5950.7929688, 4386.3300781, -5950.7929688, 4386.3300781, -10334.5771484, 10334.5771484
3: -2313.8149414, 5892.1474609, -2313.8149414, 5892.1474609, -8205.9628906, 8205.9628906
4: -6623.0805664, 4320.7236328, -6623.0805664, 4320.7236328, -10938.7910156, 10938.7919922

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1870487, upper bound: 4552.1869816
time: 0.84 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844628, upper bound: 4552.1868456
time: 0.73 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.23 seconds
IS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -4552.1869657, upper bound: 4552.2051173
IS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -4552.1844629, upper bound: 4552.1844628
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -4552.1870487, upper bound: 4552.1869816
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -4552.1844628, upper bound: 4552.1868456
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -4552.1865816, upper bound: 4552.1870487
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -4552.1844629, upper bound: 4552.1844628
IS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -4552.1870487, upper bound: 4552.1869816
IS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -4552.1844628, upper bound: 4552.1868456

## BFS IS instance: IS_B1_A1_B1

### Backsubstitution after applying IS history:
0: -2987.9899902, 2430.9243164, -2952.6821289, 2402.8015137, -5390.7915039, 5383.6064453
1: -2393.8061523, 2341.6025391, -2365.3845215, 2314.5366211, -4708.3427734, 4706.9873047
2: -3424.2626953, 2546.4648438, -3383.4587402, 2517.0407715, -5941.3027344, 5929.9223633
3: -1341.7878418, 3393.8977051, -1326.4201660, 3353.4309082, -4695.2187500, 4720.3173828
4: -3810.2302246, 2503.8750000, -3765.1281738, 2475.0305176, -6285.2602539, 6269.0029297

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1823947, upper bound: 4552.1823947
time: 0.88 seconds

## Relational analysis of IS_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1823947, upper bound: 4552.1823947
time: 0.90 seconds

## BFS IS instance: IS_B1_A1_B2

### Backsubstitution after applying IS history:
0: -2921.8796387, 2377.2465820, -4170.6796875, 3353.5466309, -6275.4262695, 6547.9262695
1: -2340.6748047, 2289.5957031, -3353.7053223, 3227.4995117, -5568.1723633, 5643.3007812
2: -3348.2663574, 2489.8530273, -4809.2104492, 3499.4179688, -6847.6840820, 7299.0634766
3: -1312.1586914, 3318.3386230, -1842.2636719, 4749.8422852, -6062.0009766, 5160.6015625
4: -3725.8732910, 2448.6198730, -5329.5825195, 3445.3452148, -7171.2187500, 7778.2021484

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B1_A1_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1823947, upper bound: 4552.1844629
time: 0.90 seconds

## Relational analysis of IS_B1_A1_B2_A2

### Relational analysis result of IS_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1823947, upper bound: 4552.1844629
time: 0.81 seconds

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -5150.8144531, 4161.5288086, -2987.9899902, 2430.9243164, -7581.7387695, 7149.5180664
1: -4128.9897461, 4006.1320801, -2393.8061523, 2341.6025391, -6470.5922852, 6399.9370117
2: -5907.2587891, 4354.6342773, -3424.2626953, 2546.4648438, -8453.7236328, 7778.8969727
3: -2297.1237793, 5849.0156250, -1341.7878418, 3393.8977051, -5691.0209961, 7190.8037109
4: -6574.7988281, 4289.6323242, -3810.2302246, 2503.8750000, -9078.6738281, 8099.8623047

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B1_A2_A1_A1

### Relational analysis result of IS_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.3927484, upper bound: 4552.1865516
time: 0.80 seconds

## Relational analysis of IS_B1_A2_A1_A2

### Relational analysis result of IS_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1727655, upper bound: 4552.1843355
time: 0.73 seconds

## BFS IS instance: IS_B1_A2_A2

### Backsubstitution after applying IS history:
0: -6400.3803711, 5147.3999023, -2921.8796387, 2377.2465820, -8777.6269531, 8069.2788086
1: -5135.4365234, 4949.7714844, -2340.6748047, 2289.5957031, -7425.0312500, 7290.4453125
2: -7353.9785156, 5379.6284180, -3348.2663574, 2489.8530273, -9843.8320312, 8727.8945312
3: -2838.2719727, 7270.3305664, -1312.1586914, 3318.3386230, -6156.6098633, 8582.4892578
4: -8166.2553711, 5302.7548828, -3725.8732910, 2448.6198730, -10614.8750000, 9028.6279297

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844629, upper bound: 4552.1848337
time: 0.76 seconds

## Relational analysis of IS_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844629, upper bound: 4552.1868456
time: 0.83 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -2979.2597656, 2424.2385254, -5150.8139648, 4161.5283203, -7140.7880859, 7575.0517578
1: -2386.7011719, 2335.1318359, -4128.9902344, 4006.1320801, -6392.8325195, 6464.1220703
2: -3414.0134277, 2539.3676758, -5907.2587891, 4354.6342773, -7768.6474609, 8446.6269531
3: -1337.9373779, 3384.0065918, -2297.1237793, 5849.0156250, -7186.9531250, 5681.1293945
4: -3798.9916992, 2496.9252930, -6574.7988281, 4289.6323242, -8088.6240234, 9071.7236328

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A1_B1_B1

### Relational analysis result of IS_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1865516, upper bound: 4551.3927484
time: 0.97 seconds

## Relational analysis of IS_B2_A1_B1_B2

### Relational analysis result of IS_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1843355, upper bound: 4552.1727655
time: 0.84 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -2920.2644043, 2376.0065918, -6400.3803711, 5147.3999023, -8067.6640625, 8776.3867188
1: -2339.3603516, 2288.3945312, -5135.4365234, 4949.7714844, -7289.1318359, 7423.8310547
2: -3346.3713379, 2488.5354004, -7353.9785156, 5379.6284180, -8726.0000000, 9842.5136719
3: -1311.4429932, 3316.5080566, -2838.2719727, 7270.3305664, -8581.7734375, 6154.7788086
4: -3723.7958984, 2447.3295898, -8166.2553711, 5302.7548828, -9026.5498047, 10613.5839844

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1848337, upper bound: 4552.1844629
time: 0.92 seconds

## Relational analysis of IS_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1848337, upper bound: 4552.1844629
time: 0.87 seconds

## BFS IS instance: IS_B2_A2_A1

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -5188.6577148, 4191.8315430, -9350.2294922, 9356.1972656
1: -4135.0488281, 4011.9240723, -4159.4165039, 4035.3022461, -8170.3510742, 8171.3408203
2: -5915.9174805, 4360.9184570, -5950.7929688, 4386.3300781, -10299.6093750, 10309.1445312
3: -2300.4028320, 5857.5903320, -2313.8149414, 5892.1474609, -8192.5507812, 8171.4052734
4: -6584.4262695, 4295.7910156, -6623.0805664, 4320.7236328, -10900.1123047, 10913.8359375

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1821376, upper bound: 4552.1845361
time: 0.73 seconds

## Relational analysis of IS_B2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1821376, upper bound: 4552.1868456
time: 0.96 seconds

## BFS IS instance: IS_B2_A2_A2

### Backsubstitution after applying IS history:
0: -6398.6328125, 5146.0449219, -5130.9306641, 4144.8867188, -10543.5185547, 10273.3105469
1: -5134.0185547, 4948.4814453, -4112.9584961, 3989.9802246, -9123.9980469, 9060.1044922
2: -7351.8583984, 5378.2460938, -5884.4077148, 4337.2500000, -11681.3779297, 11253.3291016
3: -2837.5395508, 7268.2983398, -2288.2524414, 5826.2822266, -8660.0742188, 9536.9716797
4: -8163.9833984, 5301.3891602, -6549.5268555, 4272.6474609, -12431.5634766, 11838.2109375

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B2_A2_A2_B1

### Relational analysis result of IS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844629, upper bound: 4552.1845361
time: 0.88 seconds

## Relational analysis of IS_B2_A2_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844629, upper bound: 4552.1868456
time: 0.79 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.33 seconds
IS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -4552.1823947, upper bound: 4552.1823947
IS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -4552.1823947, upper bound: 4552.1823947
IS_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -4552.1823947, upper bound: 4552.1844629
IS_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -4552.1823947, upper bound: 4552.1844629
IS_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -4551.3927484, upper bound: 4552.1865516
IS_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -4552.1727655, upper bound: 4552.1843355
IS_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -4552.1844629, upper bound: 4552.1848337
IS_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -4552.1844629, upper bound: 4552.1868456
IS_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -4552.1865516, upper bound: 4551.3927484
IS_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -4552.1843355, upper bound: 4552.1727655
IS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -4552.1848337, upper bound: 4552.1844629
IS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -4552.1848337, upper bound: 4552.1844629
IS_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -4552.1821376, upper bound: 4552.1845361
IS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -4552.1821376, upper bound: 4552.1868456
IS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -4552.1844629, upper bound: 4552.1845361
IS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -4552.1844629, upper bound: 4552.1868456

## BFS IS instance: IS_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -2952.6821289, 2402.8015137, -5355.4833984, 5355.4833984
1: -2365.3845215, 2314.5366211, -2365.3845215, 2314.5366211, -4679.9208984, 4679.9208984
2: -3383.4587402, 2517.0407715, -3383.4587402, 2517.0407715, -5900.4965820, 5900.4970703
3: -1326.4201660, 3353.4309082, -1326.4201660, 3353.4309082, -4679.8505859, 4679.8505859
4: -3765.1281738, 2475.0305176, -3765.1281738, 2475.0305176, -6240.1582031, 6240.1582031

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B1_A1_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1849741, upper bound: 4551.4338886
time: 0.77 seconds

## Relational analysis of IS_B1_A1_B1_A1_B2

### Relational analysis result of IS_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1823649, upper bound: 4552.1968862
time: 0.94 seconds

## BFS IS instance: IS_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4147.4443359, 3335.8708496, -2952.6821289, 2402.8015137, -6550.2460938, 6288.5527344
1: -3334.5939941, 3210.3688965, -2365.3845215, 2314.5366211, -5649.1308594, 5575.7534180
2: -4781.4511719, 3481.1875000, -3383.4587402, 2517.0407715, -7298.4897461, 6864.6464844
3: -1832.9420166, 4722.7758789, -1326.4201660, 3353.4309082, -5186.3725586, 6049.1962891
4: -5299.2260742, 3427.6223145, -3765.1281738, 2475.0305176, -7774.2563477, 7192.7504883

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B1_A1_B1_A2_B1

### Relational analysis result of IS_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1849741, upper bound: 4551.4338886
time: 0.84 seconds

## Relational analysis of IS_B1_A1_B1_A2_B2

### Relational analysis result of IS_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1823649, upper bound: 4552.1968862
time: 1.01 seconds

## BFS IS instance: IS_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -4147.4443359, 3335.8708496, -6288.5527344, 6550.2460938
1: -2365.3845215, 2314.5366211, -3334.5939941, 3210.3688965, -5575.7534180, 5649.1308594
2: -3383.4587402, 2517.0407715, -4781.4511719, 3481.1875000, -6864.6459961, 7298.4902344
3: -1326.4201660, 3353.4309082, -1832.9420166, 4722.7758789, -6049.1953125, 5186.3725586
4: -3765.1281738, 2475.0305176, -5299.2260742, 3427.6223145, -7192.7504883, 7774.2563477

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B1_A1_B2_A1_A1

### Relational analysis result of IS_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4338886, upper bound: 4552.1840745
time: 0.80 seconds

## Relational analysis of IS_B1_A1_B2_A1_A2

### Relational analysis result of IS_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1792395, upper bound: 4552.1809381
time: 0.85 seconds

## BFS IS instance: IS_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4267.7421875, 3427.3356934, -4267.7421875, 3427.3356934, -7695.0771484, 7695.0771484
1: -3433.8293457, 3298.9086914, -3433.8293457, 3298.9086914, -6732.7377930, 6732.7377930
2: -4925.3828125, 3575.4016113, -4925.3828125, 3575.4016113, -8500.7832031, 8500.7841797
3: -1881.0106201, 4863.0942383, -1881.0106201, 4863.0942383, -6744.1049805, 6744.1049805
4: -5456.5566406, 3519.1264648, -5456.5566406, 3519.1264648, -8975.6835938, 8975.6835938

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B1_A1_B2_A2_A1

### Relational analysis result of IS_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4338886, upper bound: 4552.1834800
time: 0.86 seconds

## Relational analysis of IS_B1_A1_B2_A2_A2

### Relational analysis result of IS_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1792395, upper bound: 4552.1801747
time: 0.79 seconds

## BFS IS instance: IS_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -5018.7734375, 4059.0698242, -2915.0573730, 2373.4636230, -7392.2368164, 6974.1269531
1: -4022.9997559, 3907.8754883, -2335.2309570, 2286.3322754, -6309.3320312, 6243.1064453
2: -5757.0004883, 4246.8969727, -3340.8876953, 2486.0581055, -8243.0576172, 7587.7836914
3: -2240.0793457, 5704.1083984, -1310.0583496, 3312.5886230, -5552.6674805, 7014.1669922
4: -6407.6645508, 4183.2099609, -3717.9489746, 2444.1931152, -8851.8574219, 7901.1591797

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_A1_A1

### Relational analysis result of IS_B1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.1850559, upper bound: 4552.1864669
time: 0.79 seconds

## Relational analysis of IS_B1_A2_A1_A1_A2

### Relational analysis result of IS_B1_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.3927484, upper bound: 4552.1856144
time: 0.83 seconds

## BFS IS instance: IS_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -5064.5874023, 4093.4934082, -2955.5649414, 2405.1352539, -7469.7226562, 7049.0585938
1: -4059.2145996, 3940.3737793, -2367.6147461, 2316.7351074, -6375.9497070, 6307.9882812
2: -5807.2431641, 4283.2216797, -3386.7673340, 2519.2219238, -8326.4648438, 7669.9892578
3: -2259.2775879, 5751.7597656, -1327.2540283, 3357.3959961, -5616.6728516, 7079.0136719
4: -6464.6357422, 4219.6044922, -3768.9741211, 2477.1035156, -8941.7392578, 7988.5786133

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_A2_A1

### Relational analysis result of IS_B1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1481205, upper bound: 4552.1843075
time: 0.85 seconds

## Relational analysis of IS_B1_A2_A1_A2_A2

### Relational analysis result of IS_B1_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1727655, upper bound: 4552.1830394
time: 0.99 seconds

## BFS IS instance: IS_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -6383.2167969, 5134.0083008, -2952.6821289, 2402.8015137, -8786.0185547, 8086.6904297
1: -5121.5708008, 4936.9619141, -2365.3845215, 2314.5366211, -7436.1074219, 7302.3461914
2: -7333.5361328, 5365.8393555, -3383.4587402, 2517.0407715, -9850.5761719, 8749.2978516
3: -2830.9841309, 7250.5678711, -1326.4201660, 3353.4309082, -6184.4150391, 8576.9882812
4: -8144.1147461, 5289.1640625, -3765.1281738, 2475.0305176, -10619.1435547, 9054.2919922

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A2_B1_A1

### Relational analysis result of IS_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9471326, upper bound: 4552.1845996
time: 0.86 seconds

## Relational analysis of IS_B1_A2_A2_B1_A2

### Relational analysis result of IS_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844628, upper bound: 4552.1838102
time: 0.88 seconds

## BFS IS instance: IS_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -6463.7666016, 5196.3979492, -4267.7421875, 3427.3356934, -9891.1025391, 9464.1406250
1: -5187.1523438, 4996.6484375, -3433.8293457, 3298.9086914, -8486.0605469, 8430.4765625
2: -7429.5161133, 5430.0644531, -4925.3828125, 3575.4016113, -11004.9169922, 10355.4433594
3: -2864.9028320, 7343.1010742, -1881.0106201, 4863.0942383, -7723.8178711, 9224.1103516
4: -8247.9746094, 5352.4482422, -5456.5566406, 3519.1264648, -11767.1005859, 10809.0048828

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A2_B2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9471326, upper bound: 4552.1862814
time: 0.81 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844629, upper bound: 4552.1856969
time: 0.82 seconds

## BFS IS instance: IS_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -2906.3190918, 2366.7399902, -5018.7734375, 4059.0698242, -6965.3886719, 7385.5136719
1: -2328.1210938, 2279.8254395, -4022.9997559, 3907.8754883, -6235.9965820, 6302.8251953
2: -3330.6279297, 2478.9125977, -5757.0004883, 4246.8969727, -7577.5244141, 8235.9121094
3: -1306.1846924, 3302.6791992, -2240.0793457, 5704.1083984, -7010.2919922, 5542.7587891
4: -3706.6977539, 2437.2009277, -6407.6645508, 4183.2099609, -7889.9077148, 8844.8652344

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_B1_B1

### Relational analysis result of IS_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1861172, upper bound: 4551.1850559
time: 0.82 seconds

## Relational analysis of IS_B2_A1_B1_B1_B2

### Relational analysis result of IS_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1856144, upper bound: 4551.3927484
time: 0.81 seconds

## BFS IS instance: IS_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -2948.4018555, 2399.6369629, -5064.5874023, 4093.4934082, -7041.8955078, 7464.2241211
1: -2361.7858887, 2311.4140625, -4059.2145996, 3940.3737793, -6302.1596680, 6370.6289062
2: -3378.3593750, 2513.3820801, -5807.2431641, 4283.2216797, -7661.5810547, 8320.6250000
3: -1324.0845947, 3349.2756348, -2259.2775879, 5751.7597656, -7075.8442383, 5608.5517578
4: -3759.7526855, 2471.3869629, -6464.6357422, 4219.6044922, -7979.3574219, 8936.0224609

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_B2_B1

### Relational analysis result of IS_B2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1843075, upper bound: 4552.1481205
time: 0.87 seconds

## Relational analysis of IS_B2_A1_B1_B2_B2

### Relational analysis result of IS_B2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1830394, upper bound: 4552.1727655
time: 0.83 seconds

## BFS IS instance: IS_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -2946.5974121, 2398.1357422, -6383.2167969, 5134.0083008, -8080.6054688, 8781.3525391
1: -2360.4313965, 2310.0227051, -5121.5708008, 4936.9619141, -7297.3935547, 7431.5927734
2: -3376.3110352, 2512.0881348, -7333.5361328, 5365.8393555, -8742.1484375, 9845.6240234
3: -1323.7344971, 3346.5322266, -2830.9841309, 7250.5678711, -8574.3027344, 6177.5166016
4: -3757.2946777, 2470.1801758, -8144.1147461, 5289.1640625, -9046.4580078, 10614.2949219

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B2_A1_B1

### Relational analysis result of IS_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1845996, upper bound: 4551.9471326
time: 0.87 seconds

## Relational analysis of IS_B2_A1_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1838102, upper bound: 4552.1844629
time: 0.85 seconds

## BFS IS instance: IS_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4267.7421875, 3427.3356934, -6463.7666016, 5196.3984375, -9464.1406250, 9891.1025391
1: -3433.8293457, 3298.9086914, -5187.1523438, 4996.6484375, -8430.4765625, 8486.0605469
2: -4925.3828125, 3575.4016113, -7429.5161133, 5430.0644531, -10355.4433594, 11004.9169922
3: -1881.0106201, 4863.0942383, -2864.9028320, 7343.1000977, -9224.1103516, 7723.8173828
4: -5456.5566406, 3519.1264648, -8247.9746094, 5352.4477539, -10809.0039062, 11767.1015625

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A1_B2_A2_B1

### Relational analysis result of IS_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1823852, upper bound: 4551.4591523
time: 0.98 seconds

## Relational analysis of IS_B2_A1_B2_A2_B2

### Relational analysis result of IS_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1826962, upper bound: 4552.1801552
time: 0.84 seconds

## BFS IS instance: IS_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -5158.3984375, 4167.5395508, -9325.9375000, 9325.9375000
1: -4135.0488281, 4011.9240723, -4135.0488281, 4011.9240723, -8146.9726562, 8146.9726562
2: -5915.9174805, 4360.9184570, -5915.9174805, 4360.9184570, -10274.1767578, 10274.1767578
3: -2300.4028320, 5857.5903320, -2300.4028320, 5857.5903320, -8157.9863281, 8157.9863281
4: -6584.4262695, 4295.7910156, -6584.4262695, 4295.7910156, -10875.1572266, 10875.1562500

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B2_A2_A1_B1_A1

### Relational analysis result of IS_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.3927484, upper bound: 4552.1786715
time: 0.83 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1727655, upper bound: 4552.1765734
time: 0.92 seconds

## BFS IS instance: IS_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -6380.2011719, 5131.6547852, -10286.4775391, 10547.7402344
1: -4135.0488281, 4011.9240723, -5119.1357422, 4934.7089844, -9068.4882812, 9131.0595703
2: -5915.9174805, 4360.9184570, -7329.9541016, 5363.4111328, -11270.1064453, 11683.6611328
3: -2300.4028320, 5857.5903320, -2829.7011719, 7247.1020508, -9528.3652344, 8683.8417969
4: -6584.4262695, 4295.7910156, -8140.2290039, 5286.7724609, -11858.6123047, 12431.2236328

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B2_A2_A1_B2_A1

### Relational analysis result of IS_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.3927484, upper bound: 4552.1865516
time: 0.84 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2

### Relational analysis result of IS_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1727655, upper bound: 4552.1843355
time: 0.79 seconds

## BFS IS instance: IS_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -6380.2011719, 5131.6547852, -5158.3984375, 4167.5395508, -10547.7402344, 10286.4775391
1: -5119.1357422, 4934.7089844, -4135.0488281, 4011.9240723, -9131.0595703, 9068.4882812
2: -7329.9541016, 5363.4111328, -5915.9174805, 4360.9184570, -11683.6611328, 11270.1064453
3: -2829.7011719, 7247.1020508, -2300.4028320, 5857.5903320, -8683.8417969, 9528.3642578
4: -8140.2290039, 5286.7724609, -6584.4262695, 4295.7910156, -12431.2236328, 11858.6123047

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A2_A2_B1_B1

### Relational analysis result of IS_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1840746, upper bound: 4551.3965813
time: 0.81 seconds

## Relational analysis of IS_B2_A2_A2_B1_B2

### Relational analysis result of IS_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1809381, upper bound: 4552.1763418
time: 0.99 seconds

## BFS IS instance: IS_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -6465.6679688, 5197.9145508, -6465.6679688, 5197.9145508, -11659.2441406, 11659.2441406
1: -5188.6606445, 4998.1103516, -5188.6606445, 4998.1103516, -10185.4218750, 10185.4208984
2: -7431.6689453, 5431.6601562, -7431.6689453, 5431.6601562, -12847.5341797, 12847.5341797
3: -2865.7429199, 7345.2402344, -2865.7429199, 7345.2402344, -10186.5107422, 10186.5107422
4: -8250.3779297, 5354.0131836, -8250.3779297, 5354.0131836, -13591.3330078, 13591.3330078

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B2_A2_A2_B2_A1

### Relational analysis result of IS_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4606772, upper bound: 4552.1861113
time: 0.90 seconds

## Relational analysis of IS_B2_A2_A2_B2_A2

### Relational analysis result of IS_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1809381, upper bound: 4552.1840125
time: 0.88 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.61 seconds
IS_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -4552.1849741, upper bound: 4551.4338886
IS_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -4552.1823649, upper bound: 4552.1968862
IS_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -4552.1849741, upper bound: 4551.4338886
IS_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -4552.1823649, upper bound: 4552.1968862
IS_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -4551.4338886, upper bound: 4552.1840745
IS_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -4552.1792395, upper bound: 4552.1809381
IS_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -4551.4338886, upper bound: 4552.1834800
IS_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -4552.1792395, upper bound: 4552.1801747
IS_B1_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -4551.1850559, upper bound: 4552.1864669
IS_B1_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -4551.3927484, upper bound: 4552.1856144
IS_B1_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -4552.1481205, upper bound: 4552.1843075
IS_B1_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -4552.1727655, upper bound: 4552.1830394
IS_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -4551.9471326, upper bound: 4552.1845996
IS_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -4552.1844628, upper bound: 4552.1838102
IS_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -4551.9471326, upper bound: 4552.1862814
IS_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -4552.1844629, upper bound: 4552.1856969
IS_B2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -4552.1861172, upper bound: 4551.1850559
IS_B2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -4552.1856144, upper bound: 4551.3927484
IS_B2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -4552.1843075, upper bound: 4552.1481205
IS_B2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -4552.1830394, upper bound: 4552.1727655
IS_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -4552.1845996, upper bound: 4551.9471326
IS_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -4552.1838102, upper bound: 4552.1844629
IS_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -4552.1823852, upper bound: 4551.4591523
IS_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -4552.1826962, upper bound: 4552.1801552
IS_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -4551.3927484, upper bound: 4552.1786715
IS_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -4552.1727655, upper bound: 4552.1765734
IS_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -4551.3927484, upper bound: 4552.1865516
IS_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -4552.1727655, upper bound: 4552.1843355
IS_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -4552.1840746, upper bound: 4551.3965813
IS_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -4552.1809381, upper bound: 4552.1763418
IS_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -4551.4606772, upper bound: 4552.1861113
IS_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 0, lower bound: -4552.1809381, upper bound: 4552.1840125

## BFS IS instance: IS_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -2879.6191406, 2345.2102051, -2829.3090820, 2307.1660156, -5186.7851562, 5174.5195312
1: -2306.7031250, 2259.1445312, -2266.5866699, 2223.4016113, -4530.1044922, 4525.7314453
2: -3299.9145508, 2456.4946289, -3243.7138672, 2416.2275391, -5716.1401367, 5700.2084961
3: -1294.6270752, 3271.9492188, -1272.9768066, 3218.7609863, -4513.3876953, 4544.9248047
4: -3672.6706543, 2415.2148438, -3609.5827637, 2374.6940918, -6047.3637695, 6024.7978516

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4372550, upper bound: 4551.4372550
time: 0.84 seconds

## Relational analysis of IS_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4372550, upper bound: 4551.4372550
time: 0.93 seconds

## BFS IS instance: IS_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -2920.2929688, 2377.0285645, -2854.9377441, 2325.1379395, -5245.4306641, 5231.9663086
1: -2339.2177734, 2289.6936035, -2286.3728027, 2239.5864258, -4578.8041992, 4576.0664062
2: -3345.9919434, 2489.8168945, -3270.2888184, 2435.0783691, -5781.0703125, 5760.1054688
3: -1311.8900146, 3316.9638672, -1282.8604736, 3243.4042969, -4555.2944336, 4599.8242188
4: -3723.9130859, 2448.2719727, -3640.6374512, 2394.5166016, -6118.4277344, 6088.9091797

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4372550, upper bound: 4552.2012449
time: 0.88 seconds

## Relational analysis of IS_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4372550, upper bound: 4552.2012449
time: 0.87 seconds

## BFS IS instance: IS_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4071.2966309, 3275.0969238, -2829.3090820, 2307.1660156, -6378.4628906, 6104.4062500
1: -3273.5939941, 3152.2126465, -2266.5866699, 2223.4016113, -5496.9946289, 5418.7993164
2: -4694.7348633, 3417.2734375, -3243.7138672, 2416.2275391, -7110.9599609, 6660.9868164
3: -1798.9521484, 4638.1406250, -1272.9768066, 3218.7609863, -5017.7128906, 5911.1171875
4: -5203.2465820, 3364.0502930, -3609.5827637, 2374.6940918, -7577.9404297, 6973.6323242

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4634052, upper bound: 4551.4338886
time: 0.85 seconds

## Relational analysis of IS_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4634052, upper bound: 4551.4338886
time: 0.97 seconds

## BFS IS instance: IS_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4115.3657227, 3310.0168457, -2854.9377441, 2325.1379395, -6440.5034180, 6164.9545898
1: -3308.6979980, 3185.4902344, -2286.3728027, 2239.5864258, -5548.2841797, 5471.8632812
2: -4744.3916016, 3453.9697266, -3270.2888184, 2435.0783691, -7179.4692383, 6724.2583008
3: -1818.4583740, 4686.3808594, -1282.8604736, 3243.4042969, -5061.8623047, 5969.2412109
4: -5258.4204102, 3400.7702637, -3640.6374512, 2394.5166016, -7652.9365234, 7041.4077148

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4634052, upper bound: 4552.1968863
time: 1.04 seconds

## Relational analysis of IS_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4634052, upper bound: 4552.1968863
time: 1.01 seconds

## BFS IS instance: IS_B1_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -2829.3090820, 2307.1660156, -4071.2966309, 3275.0969238, -6104.4062500, 6378.4628906
1: -2266.5866699, 2223.4016113, -3273.5939941, 3152.2126465, -5418.7988281, 5496.9946289
2: -3243.7138672, 2416.2275391, -4694.7348633, 3417.2734375, -6660.9868164, 7110.9604492
3: -1272.9768066, 3218.7609863, -1798.9521484, 4638.1406250, -5911.1171875, 5017.7128906
4: -3609.5827637, 2374.6940918, -5203.2465820, 3364.0502930, -6973.6323242, 7577.9404297

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B1_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B1_A1_B2_A1_A1_B1

### Relational analysis result of IS_B1_A1_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4338886, upper bound: 4551.4634052
time: 0.81 seconds

## Relational analysis of IS_B1_A1_B2_A1_A1_B2

### Relational analysis result of IS_B1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4338886, upper bound: 4552.1839260
time: 0.81 seconds

## BFS IS instance: IS_B1_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -2854.9377441, 2325.1379395, -4115.3657227, 3310.0168457, -6164.9545898, 6440.5034180
1: -2286.3728027, 2239.5864258, -3308.6979980, 3185.4902344, -5471.8632812, 5548.2841797
2: -3270.2888184, 2435.0783691, -4744.3916016, 3453.9697266, -6724.2583008, 7179.4692383
3: -1282.8604736, 3243.4042969, -1818.4583740, 4686.3808594, -5969.2412109, 5061.8627930
4: -3640.6374512, 2394.5166016, -5258.4204102, 3400.7702637, -7041.4077148, 7652.9360352

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B1_A1_B2_A1_A2_B1

### Relational analysis result of IS_B1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1968863, upper bound: 4551.4634052
time: 0.89 seconds

## Relational analysis of IS_B1_A1_B2_A1_A2_B2

### Relational analysis result of IS_B1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1968863, upper bound: 4552.1839260
time: 0.86 seconds

## BFS IS instance: IS_B1_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -4175.5634766, 3350.7539062, -4193.5429688, 3368.0146484, -7543.5776367, 7544.2968750
1: -3362.1945801, 3227.0385742, -3374.6528320, 3242.1308594, -6604.3251953, 6601.6914062
2: -4825.2534180, 3494.4333496, -4841.1494141, 3512.9433594, -8338.1972656, 8335.5830078
3: -1836.1400146, 4765.6860352, -1847.6328125, 4780.8496094, -6616.9897461, 6613.3188477
4: -5343.5141602, 3436.8771973, -5363.2656250, 3456.8146973, -8800.3281250, 8800.1425781

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B1_A1_B2_A2_A1_B1

### Relational analysis result of IS_B1_A1_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4606772, upper bound: 4551.4591523
time: 0.86 seconds

## Relational analysis of IS_B1_A1_B2_A2_A1_B2

### Relational analysis result of IS_B1_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4606770, upper bound: 4552.1801747
time: 0.73 seconds

## BFS IS instance: IS_B1_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -4186.9238281, 3361.6271973, -4240.6083984, 3405.2478027, -7592.1718750, 7602.2353516
1: -3369.0449219, 3235.5971680, -3412.0769043, 3277.6491699, -6646.6943359, 6647.6738281
2: -4832.7299805, 3506.0024414, -4894.2905273, 3552.0280762, -8384.7568359, 8400.2929688
3: -1843.7967529, 4772.0375977, -1868.4433594, 4832.5170898, -6676.3134766, 6640.4809570
4: -5354.3334961, 3450.2895508, -5422.2749023, 3495.9704590, -8850.3037109, 8872.5634766

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B1_A1_B2_A2_A2_B1

### Relational analysis result of IS_B1_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1809380, upper bound: 4551.4591523
time: 0.88 seconds

## Relational analysis of IS_B1_A1_B2_A2_A2_B2

### Relational analysis result of IS_B1_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1809380, upper bound: 4552.1801747
time: 0.77 seconds

## BFS IS instance: IS_B1_A2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -4470.4956055, 3608.4084473, -2634.3149414, 2143.5644531, -6614.0600586, 6242.7236328
1: -3585.3620605, 3474.1730957, -2110.0585938, 2065.5891113, -5650.9492188, 5584.2299805
2: -5131.9487305, 3775.4865723, -3019.7097168, 2244.5407715, -7376.4892578, 6795.1958008
3: -1992.5726318, 5082.0678711, -1181.3743896, 2995.8933105, -4988.4658203, 6263.4414062
4: -5710.5449219, 3719.0471191, -3359.4008789, 2205.9616699, -7916.5058594, 7078.4482422

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A1_A1_A1_B1

### Relational analysis result of IS_B1_A2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.1837112, upper bound: 4552.1824194
time: 0.76 seconds

## Relational analysis of IS_B1_A2_A1_A1_A1_B2

### Relational analysis result of IS_B1_A2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.1844455, upper bound: 4552.1864201
time: 0.76 seconds

## BFS IS instance: IS_B1_A2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -4963.0151367, 4013.7075195, -2882.9450684, 2347.5964355, -7310.6098633, 6896.6523438
1: -3978.1394043, 3864.0932617, -2309.3212891, 2261.4448242, -6239.5839844, 6173.4145508
2: -5692.6054688, 4199.1704102, -3303.7304688, 2458.9936523, -8151.5971680, 7502.9003906
3: -2214.4243164, 5639.9697266, -1295.6516113, 3275.5612793, -5489.9848633, 6935.6210938
4: -6336.1835938, 4136.2622070, -3676.7395020, 2417.4934082, -8753.6767578, 7813.0019531

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A1_A1_A2_B1

### Relational analysis result of IS_B1_A2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.3916214, upper bound: 4552.1813944
time: 0.81 seconds

## Relational analysis of IS_B1_A2_A1_A1_A2_B2

### Relational analysis result of IS_B1_A2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.3922844, upper bound: 4552.1854630
time: 1.09 seconds

## BFS IS instance: IS_B1_A2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -4503.3085938, 3633.1252441, -2675.6860352, 2175.8579102, -6679.1665039, 6308.8115234
1: -3611.3684082, 3497.2319336, -2143.1535645, 2096.5102539, -5707.8779297, 5640.3852539
2: -5167.6791992, 3801.7480469, -3066.5961914, 2278.2812500, -7445.9599609, 6868.3432617
3: -2006.5467529, 5115.7807617, -1198.9927979, 3041.4123535, -5047.9589844, 6314.7734375
4: -5751.3276367, 3745.5878906, -3411.5895996, 2239.6188965, -7990.9458008, 7157.1772461

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A1_A2_A1_B1

### Relational analysis result of IS_B1_A2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1471731, upper bound: 4552.1802639
time: 0.83 seconds

## Relational analysis of IS_B1_A2_A1_A2_A1_B2

### Relational analysis result of IS_B1_A2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1477037, upper bound: 4552.1842568
time: 0.85 seconds

## BFS IS instance: IS_B1_A2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -5008.4155273, 4047.7583008, -2923.2517090, 2379.0974121, -7387.5126953, 6971.0097656
1: -4013.9729004, 3896.2609863, -2341.5317383, 2291.6801758, -6305.6533203, 6237.7929688
2: -5742.3564453, 4235.1704102, -3349.3566895, 2491.9855957, -8234.3417969, 7584.5268555
3: -2233.5141602, 5687.1845703, -1312.7652588, 3320.0935059, -5553.6074219, 6999.9497070
4: -6392.6396484, 4172.3242188, -3727.4973145, 2450.2375488, -8842.8769531, 7899.8212891

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B1_A2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A1_A2_A2_B1

### Relational analysis result of IS_B1_A2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1718813, upper bound: 4552.1787537
time: 0.81 seconds

## Relational analysis of IS_B1_A2_A1_A2_A2_B2

### Relational analysis result of IS_B1_A2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1723977, upper bound: 4552.1828348
time: 0.81 seconds

## BFS IS instance: IS_B1_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5883.0532227, 4712.4941406, -2671.1254883, 2172.1181641, -8055.1708984, 7383.6196289
1: -4721.4848633, 4532.3344727, -2139.6044922, 2092.9387207, -6814.4233398, 6671.9389648
2: -6761.7001953, 4923.9184570, -3061.3686523, 2274.6315918, -9036.3320312, 7985.2866211
3: -2597.5878906, 6675.6420898, -1197.4808350, 3035.2807617, -5632.8671875, 7870.9311523
4: -7506.8374023, 4854.1186523, -3405.6518555, 2236.1850586, -9743.0195312, 8259.7695312

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B1_A2_A2_B1_A1_A1

### Relational analysis result of IS_B1_A2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.0215226, upper bound: 4552.2023347
time: 1.80 seconds

## Relational analysis of IS_B1_A2_A2_B1_A1_A2

### Relational analysis result of IS_B1_A2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9468034, upper bound: 4552.2001859
time: 0.90 seconds

## BFS IS instance: IS_B1_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6268.9477539, 5045.2504883, -2921.8789062, 2377.9321289, -8646.8798828, 7967.1289062
1: -5029.4248047, 4851.1342773, -2340.5214844, 2290.6164551, -7320.0405273, 7191.6552734
2: -7200.5585938, 5273.4228516, -3347.8095703, 2491.0283203, -9691.5869141, 8621.2324219
3: -2782.2995605, 7120.6430664, -1312.5924072, 3317.8781738, -6100.1767578, 8433.2353516
4: -7997.7241211, 5197.9892578, -3725.5930176, 2449.3652344, -10447.0869141, 8923.5820312

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B1_A2_A2_B1_A2_A1

### Relational analysis result of IS_B1_A2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4634052, upper bound: 4552.2014445
time: 0.76 seconds

## Relational analysis of IS_B1_A2_A2_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1764789, upper bound: 4552.1988784
time: 0.82 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5975.7158203, 4784.3911133, -4004.6762695, 3206.4106445, -9182.1240234, 8789.0654297
1: -4797.0288086, 4601.2744141, -3224.3349609, 3088.2563477, -7885.2851562, 7825.6088867
2: -6871.9663086, 4997.9458008, -4626.9672852, 3343.3908691, -10215.3544922, 9624.9023438
3: -2636.5402832, 6782.5122070, -1755.0847168, 4565.8627930, -7197.5424805, 8532.5332031
4: -7626.3266602, 4926.9892578, -5123.0883789, 3287.5991211, -10913.9257812, 10050.0781250

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B1_A2_A2_B2_A1_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.0174780, upper bound: 4552.1858459
time: 0.85 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_A2

### Relational analysis result of IS_B1_A2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.9430535, upper bound: 4552.1839247
time: 0.77 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6359.6411133, 5116.0634766, -4218.0156250, 3389.5805664, -9749.2216797, 9334.0791016
1: -5103.2060547, 4918.9052734, -3393.8149414, 3262.1633301, -8365.3671875, 8312.7197266
2: -7308.8110352, 5346.4379883, -4867.5170898, 3536.0031738, -10844.8144531, 10213.9531250
3: -2820.8696289, 7225.2260742, -1860.0163574, 4806.6772461, -7621.8349609, 9084.9912109
4: -8114.6435547, 5269.8906250, -5392.4360352, 3480.2397461, -11594.8828125, 10662.3261719

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B1_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B1_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B1_A2_A2_B2_A2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4606770, upper bound: 4552.1852835
time: 0.75 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1809380, upper bound: 4552.1827259
time: 0.85 seconds

## BFS IS instance: IS_B2_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -2625.4511719, 2136.7424316, -4470.4956055, 3608.4084473, -6233.8593750, 6607.2382812
1: -2102.8569336, 2059.0256348, -3585.3620605, 3474.1730957, -5577.0288086, 5644.3867188
2: -3009.3159180, 2237.3200684, -5131.9487305, 3775.4865723, -6784.8012695, 7369.2685547
3: -1177.4053955, 2985.9760742, -1992.5726318, 5082.0678711, -6259.4726562, 4978.5488281
4: -3347.9682617, 2198.8400879, -5710.5449219, 3719.0471191, -7067.0141602, 7909.3837891

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A1_B1_B1_B1_A1

### Relational analysis result of IS_B2_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1819789, upper bound: 4551.1837112
time: 0.81 seconds

## Relational analysis of IS_B2_A1_B1_B1_B1_A2

### Relational analysis result of IS_B2_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1864201, upper bound: 4551.1844455
time: 0.93 seconds

## BFS IS instance: IS_B2_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -2874.2253418, 2340.8901367, -4963.0151367, 4013.7075195, -6887.9326172, 7303.9052734
1: -2302.2248535, 2254.9553223, -3978.1394043, 3864.0932617, -6166.3183594, 6233.0947266
2: -3293.4895020, 2451.8662109, -5692.6054688, 4199.1704102, -7492.6601562, 8144.4707031
3: -1291.7873535, 3265.7058105, -2214.4243164, 5639.9697266, -6931.7568359, 5480.1284180
4: -3665.5097656, 2410.5192871, -6336.1835938, 4136.2622070, -7801.7719727, 8746.7031250

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A1_B1_B1_B2_A1

### Relational analysis result of IS_B2_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1813410, upper bound: 4551.3916214
time: 0.99 seconds

## Relational analysis of IS_B2_A1_B1_B1_B2_A2

### Relational analysis result of IS_B2_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1854630, upper bound: 4551.3922844
time: 0.98 seconds

## BFS IS instance: IS_B2_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -2668.4599609, 2170.3237305, -4503.3085938, 3633.1252441, -6301.5849609, 6673.6323242
1: -2137.2829590, 2091.1835938, -3611.3684082, 3497.2319336, -5634.5146484, 5702.5517578
2: -3058.1250000, 2272.4289551, -5167.6791992, 3801.7480469, -6859.8730469, 7440.1079102
3: -1195.7789307, 3033.3356934, -2006.5467529, 5115.7807617, -6311.5595703, 5039.8823242
4: -3402.2739258, 2233.8437500, -5751.3276367, 3745.5878906, -7147.8613281, 7985.1713867

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A1_B1_B2_B1_A1

### Relational analysis result of IS_B2_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1802639, upper bound: 4552.1471731
time: 0.80 seconds

## Relational analysis of IS_B2_A1_B1_B2_B1_A2

### Relational analysis result of IS_B2_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1842568, upper bound: 4552.1477037
time: 0.85 seconds

## BFS IS instance: IS_B2_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -2916.1088867, 2373.6201172, -5008.4155273, 4047.7583008, -6963.8671875, 7382.0356445
1: -2335.7185059, 2286.3789062, -4013.9729004, 3896.2609863, -6231.9794922, 6300.3515625
2: -3340.9707031, 2486.1682129, -5742.3564453, 4235.1704102, -7576.1411133, 8228.5244141
3: -1309.6075439, 3311.9978027, -2233.5141602, 5687.1845703, -6996.7919922, 5545.5117188
4: -3718.3015137, 2444.5432129, -6392.6396484, 4172.3242188, -7890.6259766, 8837.1826172

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B2_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A1_B1_B2_B2_A1

### Relational analysis result of IS_B2_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1787537, upper bound: 4552.1718813
time: 0.88 seconds

## Relational analysis of IS_B2_A1_B1_B2_B2_A2

### Relational analysis result of IS_B2_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1828348, upper bound: 4552.1723977
time: 0.95 seconds

## BFS IS instance: IS_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -2664.9875488, 2167.4208984, -5883.0502930, 4712.4916992, -7377.4790039, 8050.4707031
1: -2134.6171875, 2088.4191895, -4721.4824219, 4532.3325195, -6666.9492188, 6809.9013672
2: -3054.1696777, 2269.6667480, -6761.6967773, 4923.9160156, -7978.0854492, 9031.3623047
3: -1194.7595215, 3028.4240723, -2597.5866699, 6675.6396484, -7868.2080078, 5626.0097656
4: -3397.7399902, 2231.2827148, -7506.8334961, 4854.1162109, -8251.8544922, 9738.1162109

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A1_B2_A1_B1_B1

### Relational analysis result of IS_B2_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2023347, upper bound: 4551.0215226
time: 0.76 seconds

## Relational analysis of IS_B2_A1_B2_A1_B1_B2

### Relational analysis result of IS_B2_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2001859, upper bound: 4551.9468034
time: 0.90 seconds

## BFS IS instance: IS_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -2915.8085938, 2373.2844238, -6268.9477539, 5045.2504883, -7961.0590820, 8642.2324219
1: -2335.5822754, 2286.1203613, -5029.4248047, 4851.1342773, -7186.7158203, 7315.5434570
2: -3340.6809082, 2486.0957031, -7200.5585938, 5273.4228516, -8614.1035156, 9686.6533203
3: -1309.9174805, 3310.9985352, -2782.2995605, 7120.6430664, -8430.5595703, 6093.2978516
4: -3717.7807617, 2444.5356445, -7997.7241211, 5197.9892578, -8915.7695312, 10442.2597656

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A1_B2_A1_B2_B1

### Relational analysis result of IS_B2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2000346, upper bound: 4551.4634052
time: 0.77 seconds

## Relational analysis of IS_B2_A1_B2_A1_B2_B2

### Relational analysis result of IS_B2_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1968863, upper bound: 4552.1839260
time: 0.92 seconds

## BFS IS instance: IS_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4193.5429688, 3368.0146484, -6361.4199219, 5115.4648438, -9309.0078125, 9729.4335938
1: -3374.6528320, 3242.1308594, -5107.0083008, 4919.5571289, -8294.2099609, 8349.1386719
2: -4841.1494141, 3512.9433594, -7317.0712891, 5344.0668945, -10184.7851562, 10830.0117188
3: -1847.6328125, 4780.8496094, -2818.8339844, 7234.5327148, -9082.1660156, 7591.4843750
4: -5363.2656250, 3456.8146973, -8120.7622070, 5266.9775391, -10630.2431641, 11577.5771484

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B2_A2_B1_B1

### Relational analysis result of IS_B2_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1862377, upper bound: 4551.0174780
time: 1.44 seconds

## Relational analysis of IS_B2_A1_B2_A2_B1_B2

### Relational analysis result of IS_B2_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1854866, upper bound: 4551.4591523
time: 0.85 seconds

## BFS IS instance: IS_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4240.6083984, 3405.2478027, -6386.0185547, 5133.8432617, -9374.4501953, 9791.2626953
1: -3412.0769043, 3277.6491699, -5124.7583008, 4936.2675781, -8348.3447266, 8402.4072266
2: -4894.2905273, 3552.0280762, -7340.0996094, 5364.0043945, -10258.2949219, 10892.1279297
3: -1868.4433594, 4832.5170898, -2829.9091797, 7255.4897461, -9122.4755859, 7655.0380859
4: -5422.2749023, 3495.9704590, -8149.0068359, 5287.6948242, -10709.9697266, 11644.9765625

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B2_A2_B2_B1

### Relational analysis result of IS_B2_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1841705, upper bound: 4551.9430535
time: 0.93 seconds

## Relational analysis of IS_B2_A1_B2_A2_B2_B2

### Relational analysis result of IS_B2_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1829684, upper bound: 4552.1801552
time: 0.86 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5030.1420898, 4068.0788574, -5094.6083984, 4117.0488281, -9147.1904297, 9162.5615234
1: -4032.0764160, 3916.5642090, -4083.5900879, 3963.2976074, -7995.3730469, 8000.1533203
2: -5769.9741211, 4256.2983398, -5842.6342773, 4308.1328125, -10072.7636719, 10092.8437500
3: -2244.9921875, 5716.9531250, -2272.7023926, 5785.8061523, -8028.3652344, 7989.5756836
4: -6422.0834961, 4192.4262695, -6503.3540039, 4243.6406250, -10658.0527344, 10687.5283203

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A2_A1_B1_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.3957988, upper bound: 4551.3967212
time: 0.82 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_B2

### Relational analysis result of IS_B2_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.3957988, upper bound: 4552.1765734
time: 0.90 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5074.3295898, 4101.2333984, -5128.8793945, 4144.2397461, -9218.5673828, 9230.1132812
1: -4067.0036621, 3947.8349609, -4111.1611328, 3989.4077148, -8056.4106445, 8058.9946289
2: -5818.3769531, 4291.3007812, -5881.6923828, 4336.4731445, -10152.8281250, 10168.8408203
3: -2263.4909668, 5762.7929688, -2287.4533691, 5824.2978516, -8085.7226562, 8049.7441406
4: -6477.0117188, 4227.5278320, -6546.7275391, 4271.8125000, -10744.3095703, 10767.7744141

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A2_A1_B1_A2_B1

### Relational analysis result of IS_B2_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1758774, upper bound: 4551.3967212
time: 0.94 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_B2

### Relational analysis result of IS_B2_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1758774, upper bound: 4552.1765734
time: 0.80 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5030.1420898, 4068.0788574, -6308.8081055, 5075.1040039, -10099.1044922, 10375.7851562
1: -4032.0764160, 3916.5642090, -5061.7978516, 4880.4208984, -8909.1787109, 8978.3603516
2: -5769.9741211, 4256.2983398, -7248.3945312, 5304.0756836, -11062.1826172, 11494.1650391
3: -2244.9921875, 5716.9531250, -2798.4653320, 7167.4350586, -9390.8095703, 8511.8925781
4: -6422.0834961, 4192.4262695, -8049.8627930, 5228.1728516, -11635.0732422, 12234.3515625

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A2_A1_B2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.3927484, upper bound: 4551.4634914
time: 0.94 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.3927484, upper bound: 4552.1843354
time: 0.79 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5074.3295898, 4101.2333984, -6350.9726562, 5108.3315430, -10178.8906250, 10452.2060547
1: -4067.0036621, 3947.8349609, -5095.5781250, 4912.2250977, -8977.6572266, 9043.4121094
2: -5818.3769531, 4291.3007812, -7296.1425781, 5338.8935547, -11147.9921875, 11578.6455078
3: -2263.4909668, 5762.7929688, -2816.7231445, 7214.0844727, -9456.3125000, 8575.1708984
4: -6477.0117188, 4227.5278320, -8102.9487305, 5262.7119141, -11726.8916016, 12324.2353516

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A2_A1_B2_A2_B1

### Relational analysis result of IS_B2_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1727655, upper bound: 4551.4634914
time: 0.76 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2_B2

### Relational analysis result of IS_B2_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1727655, upper bound: 4552.1843354
time: 0.88 seconds

## BFS IS instance: IS_B2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -6308.8081055, 5075.1040039, -5030.1420898, 4068.0788574, -10375.7841797, 10099.1044922
1: -5061.7978516, 4880.4208984, -4032.0764160, 3916.5642090, -8978.3603516, 8909.1777344
2: -7248.3945312, 5304.0756836, -5769.9741211, 4256.2983398, -11494.1650391, 11062.1826172
3: -2798.4653320, 7167.4350586, -2244.9921875, 5716.9531250, -8511.8925781, 9390.8085938
4: -8049.8627930, 5228.1728516, -6422.0834961, 4192.4262695, -12234.3515625, 11635.0732422

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B2_A2_A2_B1_B1_A1

### Relational analysis result of IS_B2_A2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4627349, upper bound: 4551.3965813
time: 0.78 seconds

## Relational analysis of IS_B2_A2_A2_B1_B1_A2

### Relational analysis result of IS_B2_A2_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4627349, upper bound: 4551.3965813
time: 0.81 seconds

## BFS IS instance: IS_B2_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -6350.9726562, 5108.3315430, -5074.3295898, 4101.2333984, -10452.2060547, 10178.8906250
1: -5095.5781250, 4912.2250977, -4067.0036621, 3947.8349609, -9043.4121094, 8977.6572266
2: -7296.1425781, 5338.8935547, -5818.3769531, 4291.3007812, -11578.6455078, 11147.9921875
3: -2816.7231445, 7214.0844727, -2263.4909668, 5762.7929688, -8575.1699219, 9456.3125000
4: -8102.9487305, 5262.7119141, -6477.0117188, 4227.5278320, -12324.2343750, 11726.8916016

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B2_A2_A2_B1_B2_A1

### Relational analysis result of IS_B2_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4627349, upper bound: 4552.1763418
time: 1.01 seconds

## Relational analysis of IS_B2_A2_A2_B1_B2_A2

### Relational analysis result of IS_B2_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4627349, upper bound: 4552.1763418
time: 0.79 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6366.1660156, 5119.2890625, -6395.9633789, 5142.6381836, -11502.4218750, 11506.3974609
1: -5110.7729492, 4923.2451172, -5132.8325195, 4945.0581055, -10052.0068359, 10050.5947266
2: -7322.4482422, 5348.0849609, -7352.1743164, 5373.6157227, -12676.8886719, 12678.7539062
3: -2820.9536133, 7239.8808594, -2835.1943359, 7267.5712891, -10060.0761719, 10049.8896484
4: -8126.7597656, 5270.9204102, -8162.2187500, 5296.6889648, -13407.6357422, 13414.5869141

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A2_A2_B2_A1_B1

### Relational analysis result of IS_B2_A2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4606770, upper bound: 4551.4630710
time: 0.86 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_B2

### Relational analysis result of IS_B2_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4606770, upper bound: 4552.1840124
time: 0.87 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6390.3413086, 5137.2963867, -6440.4824219, 5177.6611328, -11563.2167969, 11570.5810547
1: -5128.1879883, 4939.5961914, -5168.4379883, 4978.5522461, -10104.6757812, 10104.0664062
2: -7344.9951172, 5367.6391602, -7402.6870117, 5410.2905273, -12738.7343750, 12751.1259766
3: -2831.8232422, 7260.3583984, -2854.4323730, 7316.8466797, -10120.9785156, 10088.8271484
4: -8154.4711914, 5291.2568359, -8218.3173828, 5333.0590820, -13473.9052734, 13493.0185547

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A2_A2_B2_A2_B1

### Relational analysis result of IS_B2_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1809380, upper bound: 4551.4630710
time: 0.93 seconds

## Relational analysis of IS_B2_A2_A2_B2_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1809380, upper bound: 4552.1840124
time: 0.86 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.35 seconds
IS_B1_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 4.35
Output dim: 0, lower bound: -4551.4372550, upper bound: 4551.4372550
IS_B1_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 6, time: 4.35
Output dim: 0, lower bound: -4551.4372550, upper bound: 4551.4372550
IS_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4551.4372550, upper bound: 4552.2012449
IS_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4551.4372550, upper bound: 4552.2012449
IS_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 4.35
Output dim: 0, lower bound: -4551.4634052, upper bound: 4551.4338886
IS_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 4.35
Output dim: 0, lower bound: -4551.4634052, upper bound: 4551.4338886
IS_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4551.4634052, upper bound: 4552.1968863
IS_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4551.4634052, upper bound: 4552.1968863
IS_B1_A1_B2_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.35
Output dim: 0, lower bound: -4551.4338886, upper bound: 4551.4634052
IS_B1_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4551.4338886, upper bound: 4552.1839260
IS_B1_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4552.1968863, upper bound: 4551.4634052
IS_B1_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4552.1968863, upper bound: 4552.1839260
IS_B1_A1_B2_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.35
Output dim: 0, lower bound: -4551.4606772, upper bound: 4551.4591523
IS_B1_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4551.4606770, upper bound: 4552.1801747
IS_B1_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4552.1809380, upper bound: 4551.4591523
IS_B1_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4552.1809380, upper bound: 4552.1801747
IS_B1_A2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4551.1837112, upper bound: 4552.1824194
IS_B1_A2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4551.1844455, upper bound: 4552.1864201
IS_B1_A2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4551.3916214, upper bound: 4552.1813944
IS_B1_A2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4551.3922844, upper bound: 4552.1854630
IS_B1_A2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4552.1471731, upper bound: 4552.1802639
IS_B1_A2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4552.1477037, upper bound: 4552.1842568
IS_B1_A2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4552.1718813, upper bound: 4552.1787537
IS_B1_A2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4552.1723977, upper bound: 4552.1828348
IS_B1_A2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4551.0215226, upper bound: 4552.2023347
IS_B1_A2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4551.9468034, upper bound: 4552.2001859
IS_B1_A2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4551.4634052, upper bound: 4552.2014445
IS_B1_A2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4552.1764789, upper bound: 4552.1988784
IS_B1_A2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4551.0174780, upper bound: 4552.1858459
IS_B1_A2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4551.9430535, upper bound: 4552.1839247
IS_B1_A2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4551.4606770, upper bound: 4552.1852835
IS_B1_A2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4552.1809380, upper bound: 4552.1827259
IS_B2_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4552.1819789, upper bound: 4551.1837112
IS_B2_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4552.1864201, upper bound: 4551.1844455
IS_B2_A1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4552.1813410, upper bound: 4551.3916214
IS_B2_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4552.1854630, upper bound: 4551.3922844
IS_B2_A1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4552.1802639, upper bound: 4552.1471731
IS_B2_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4552.1842568, upper bound: 4552.1477037
IS_B2_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4552.1787537, upper bound: 4552.1718813
IS_B2_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4552.1828348, upper bound: 4552.1723977
IS_B2_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4552.2023347, upper bound: 4551.0215226
IS_B2_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4552.2001859, upper bound: 4551.9468034
IS_B2_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4552.2000346, upper bound: 4551.4634052
IS_B2_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4552.1968863, upper bound: 4552.1839260
IS_B2_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4552.1862377, upper bound: 4551.0174780
IS_B2_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4552.1854866, upper bound: 4551.4591523
IS_B2_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4552.1841705, upper bound: 4551.9430535
IS_B2_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4552.1829684, upper bound: 4552.1801552
IS_B2_A2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.35
Output dim: 0, lower bound: -4551.3957988, upper bound: 4551.3967212
IS_B2_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4551.3957988, upper bound: 4552.1765734
IS_B2_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4552.1758774, upper bound: 4551.3967212
IS_B2_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4552.1758774, upper bound: 4552.1765734
IS_B2_A2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.35
Output dim: 0, lower bound: -4551.3927484, upper bound: 4551.4634914
IS_B2_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4551.3927484, upper bound: 4552.1843354
IS_B2_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4552.1727655, upper bound: 4551.4634914
IS_B2_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4552.1727655, upper bound: 4552.1843354
IS_B2_A2_A2_B1_B1_A1, status: Status.VERIFIED, split count: 6, time: 4.35
Output dim: 0, lower bound: -4551.4627349, upper bound: 4551.3965813
IS_B2_A2_A2_B1_B1_A2, status: Status.VERIFIED, split count: 6, time: 4.35
Output dim: 0, lower bound: -4551.4627349, upper bound: 4551.3965813
IS_B2_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4551.4627349, upper bound: 4552.1763418
IS_B2_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4551.4627349, upper bound: 4552.1763418
IS_B2_A2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.35
Output dim: 0, lower bound: -4551.4606770, upper bound: 4551.4630710
IS_B2_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4551.4606770, upper bound: 4552.1840124
IS_B2_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4552.1809380, upper bound: 4551.4630710
IS_B2_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -4552.1809380, upper bound: 4552.1840124

## BFS IS instance: IS_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -2829.3090820, 2307.1660156, -2854.9377441, 2325.1379395, -5154.4472656, 5162.1035156
1: -2266.5866699, 2223.4016113, -2286.3728027, 2239.5864258, -4506.1728516, 4509.7744141
2: -3243.7138672, 2416.2275391, -3270.2888184, 2435.0783691, -5678.7915039, 5686.5151367
3: -1272.9768066, 3218.7609863, -1282.8604736, 3243.4042969, -4516.3803711, 4501.6215820
4: -3609.5827637, 2374.6940918, -3640.6374512, 2394.5166016, -6004.0976562, 6015.3315430

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4369387, upper bound: 4552.2007620
time: 0.80 seconds

## Relational analysis of IS_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4364525, upper bound: 4552.2007620
time: 0.74 seconds

## BFS IS instance: IS_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -2854.9377441, 2325.1379395, -2854.9377441, 2325.1379395, -5180.0756836, 5180.0756836
1: -2286.3728027, 2239.5864258, -2286.3728027, 2239.5864258, -4525.9589844, 4525.9589844
2: -3270.2888184, 2435.0783691, -3270.2888184, 2435.0783691, -5705.3666992, 5705.3666992
3: -1282.8604736, 3243.4042969, -1282.8604736, 3243.4042969, -4526.2646484, 4526.2646484
4: -3640.6374512, 2394.5166016, -3640.6374512, 2394.5166016, -6035.1542969, 6035.1538086

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4369387, upper bound: 4552.1975647
time: 0.91 seconds

## Relational analysis of IS_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4363940, upper bound: 4552.1975647
time: 1.03 seconds

## BFS IS instance: IS_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4051.2358398, 3256.9140625, -2854.9377441, 2325.1379395, -6376.3735352, 6111.8515625
1: -3259.4213867, 3136.2165527, -2286.3728027, 2239.5864258, -5499.0078125, 5422.5893555
2: -4676.0600586, 3397.9028320, -3270.2888184, 2435.0783691, -7111.1376953, 6668.1909180
3: -1787.2316895, 4620.5029297, -1282.8604736, 3243.4042969, -5030.6357422, 5903.3632812
4: -5180.4418945, 3343.5212402, -3640.6374512, 2394.5166016, -7574.9570312, 6984.1586914

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B1_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_B1_A2_B2_A1_A1

### Relational analysis result of IS_B1_A1_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4619738, upper bound: 4552.1960154
time: 0.89 seconds

## Relational analysis of IS_B1_A1_B1_A2_B2_A1_A2

### Relational analysis result of IS_B1_A1_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4626200, upper bound: 4552.1965450
time: 0.95 seconds

## BFS IS instance: IS_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4063.6889648, 3268.0424805, -2854.9377441, 2325.1379395, -6388.8256836, 6122.9804688
1: -3267.1667480, 3144.9948730, -2286.3728027, 2239.5864258, -5506.7529297, 5431.3676758
2: -4685.1044922, 3409.6069336, -3270.2888184, 2435.0783691, -7120.1821289, 6679.8950195
3: -1794.7897949, 4628.2006836, -1282.8604736, 3243.4042969, -5038.1938477, 5911.0605469
4: -5192.9995117, 3356.8891602, -3640.6374512, 2394.5166016, -7587.5146484, 6997.5263672

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B1_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_B1_A2_B2_A2_A1

### Relational analysis result of IS_B1_A1_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4619738, upper bound: 4552.1919864
time: 0.83 seconds

## Relational analysis of IS_B1_A1_B1_A2_B2_A2_A2

### Relational analysis result of IS_B1_A1_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4626200, upper bound: 4552.1925659
time: 0.74 seconds

## BFS IS instance: IS_B1_A1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -2829.3090820, 2307.1660156, -4062.8671875, 3267.4169922, -6096.7260742, 6370.0332031
1: -2266.5866699, 2223.4016113, -3266.4895020, 3144.3901367, -5410.9760742, 5489.8911133
2: -3243.7138672, 2416.2275391, -4684.1206055, 3408.9626465, -6652.6762695, 7100.3476562
3: -1272.9768066, 3218.7609863, -1794.4606934, 4627.2426758, -5900.2192383, 5013.2216797
4: -3609.5827637, 2374.6940918, -5191.9257812, 3356.2636719, -6965.8466797, 7566.6196289

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B1_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_B2_A1_A1_B2_B1

### Relational analysis result of IS_B1_A1_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4327864, upper bound: 4552.1820640
time: 0.86 seconds

## Relational analysis of IS_B1_A1_B2_A1_A1_B2_B2

### Relational analysis result of IS_B1_A1_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4334526, upper bound: 4552.1861053
time: 0.91 seconds

## BFS IS instance: IS_B1_A1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -2854.9377441, 2325.1379395, -4051.2358398, 3256.9140625, -6111.8515625, 6376.3740234
1: -2286.3728027, 2239.5864258, -3259.4213867, 3136.2165527, -5422.5893555, 5499.0078125
2: -3270.2888184, 2435.0783691, -4676.0600586, 3397.9028320, -6668.1909180, 7111.1381836
3: -1282.8604736, 3243.4042969, -1787.2316895, 4620.5029297, -5903.3632812, 5030.6357422
4: -3640.6374512, 2394.5166016, -5180.4418945, 3343.5212402, -6984.1586914, 7574.9570312

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B1_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_A1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_B2_A1_A2_B1_B1

### Relational analysis result of IS_B1_A1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1960153, upper bound: 4551.4619738
time: 0.97 seconds

## Relational analysis of IS_B1_A1_B2_A1_A2_B1_B2

### Relational analysis result of IS_B1_A1_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1965449, upper bound: 4551.4626200
time: 0.94 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.43 seconds
IS_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4551.4369387, upper bound: 4552.2007620
IS_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4551.4364525, upper bound: 4552.2007620
IS_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4551.4369387, upper bound: 4552.1975647
IS_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4551.4363940, upper bound: 4552.1975647
IS_B1_A1_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4551.4619738, upper bound: 4552.1960154
IS_B1_A1_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4551.4626200, upper bound: 4552.1965450
IS_B1_A1_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4551.4619738, upper bound: 4552.1919864
IS_B1_A1_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4551.4626200, upper bound: 4552.1925659
IS_B1_A1_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4551.4327864, upper bound: 4552.1820640
IS_B1_A1_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4551.4334526, upper bound: 4552.1861053
IS_B1_A1_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4552.1960153, upper bound: 4551.4619738
IS_B1_A1_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4552.1965449, upper bound: 4551.4626200
IS_B1_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4552.1968863, upper bound: 4552.1839260
IS_B1_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4551.4606770, upper bound: 4552.1801747
IS_B1_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4552.1809380, upper bound: 4551.4591523
IS_B1_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4552.1809380, upper bound: 4552.1801747
IS_B1_A2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4551.1837112, upper bound: 4552.1824194
IS_B1_A2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4551.1844455, upper bound: 4552.1864201
IS_B1_A2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4551.3916214, upper bound: 4552.1813944
IS_B1_A2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4551.3922844, upper bound: 4552.1854630
IS_B1_A2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4552.1471731, upper bound: 4552.1802639
IS_B1_A2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4552.1477037, upper bound: 4552.1842568
IS_B1_A2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4552.1718813, upper bound: 4552.1787537
IS_B1_A2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4552.1723977, upper bound: 4552.1828348
IS_B1_A2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4551.0215226, upper bound: 4552.2023347
IS_B1_A2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4551.9468034, upper bound: 4552.2001859
IS_B1_A2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4551.4634052, upper bound: 4552.2014445
IS_B1_A2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4552.1764789, upper bound: 4552.1988784
IS_B1_A2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4551.0174780, upper bound: 4552.1858459
IS_B1_A2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4551.9430535, upper bound: 4552.1839247
IS_B1_A2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4551.4606770, upper bound: 4552.1852835
IS_B1_A2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4552.1809380, upper bound: 4552.1827259
IS_B2_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4552.1819789, upper bound: 4551.1837112
IS_B2_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4552.1864201, upper bound: 4551.1844455
IS_B2_A1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4552.1813410, upper bound: 4551.3916214
IS_B2_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4552.1854630, upper bound: 4551.3922844
IS_B2_A1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4552.1802639, upper bound: 4552.1471731
IS_B2_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4552.1842568, upper bound: 4552.1477037
IS_B2_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4552.1787537, upper bound: 4552.1718813
IS_B2_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4552.1828348, upper bound: 4552.1723977
IS_B2_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4552.2023347, upper bound: 4551.0215226
IS_B2_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4552.2001859, upper bound: 4551.9468034
IS_B2_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4552.2000346, upper bound: 4551.4634052
IS_B2_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4552.1968863, upper bound: 4552.1839260
IS_B2_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4552.1862377, upper bound: 4551.0174780
IS_B2_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4552.1854866, upper bound: 4551.4591523
IS_B2_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4552.1841705, upper bound: 4551.9430535
IS_B2_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4552.1829684, upper bound: 4552.1801552
IS_B2_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4551.3957988, upper bound: 4552.1765734
IS_B2_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4552.1758774, upper bound: 4551.3967212
IS_B2_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4552.1758774, upper bound: 4552.1765734
IS_B2_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4551.3927484, upper bound: 4552.1843354
IS_B2_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4552.1727655, upper bound: 4551.4634914
IS_B2_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4552.1727655, upper bound: 4552.1843354
IS_B2_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4551.4627349, upper bound: 4552.1763418
IS_B2_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4551.4627349, upper bound: 4552.1763418
IS_B2_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4551.4606770, upper bound: 4552.1840124
IS_B2_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4552.1809380, upper bound: 4551.4630710
IS_B2_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4552.1809380, upper bound: 4552.1840124
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=5687.5751953125
rel_dist={0: [-4552.279487443704, 4552.279487443706]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1122.22 seconds
