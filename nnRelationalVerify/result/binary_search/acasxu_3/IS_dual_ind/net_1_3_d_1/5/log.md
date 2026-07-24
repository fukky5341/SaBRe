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
execution time: IAR + LP analysis = 1.58 + 2.09 = 3.66 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -4552.2890012, upper bound: 4552.2890012


# Binary Search by BASE starts (time budget: 1196.34 seconds, max iter: 100)

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
Binary search time: 76.74 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1119.60 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2858669, upper bound: 4552.2771766
time: 1.57 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2756825, upper bound: 4552.2756825
time: 0.86 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.57 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.57
Output dim: 0, lower bound: -4552.2858669, upper bound: 4552.2771766
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.57
Output dim: 0, lower bound: -4552.2756825, upper bound: 4552.2756825

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -2987.9899902, 2430.9243164, -3136.6748047, 2550.9006348, -5538.8906250, 5567.5991211
1: -2393.8061523, 2341.6025391, -2513.2941895, 2457.4460449, -4851.2519531, 4854.8964844
2: -3424.2626953, 2546.4648438, -3595.4755859, 2672.1398926, -6096.4023438, 6141.9404297
3: -1341.7878418, 3393.8977051, -1407.5072021, 3564.2558594, -4906.0439453, 4801.4047852
4: -3810.2302246, 2503.8750000, -3999.4067383, 2627.0041504, -6437.2343750, 6503.2817383

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2755934, upper bound: 4552.2755934
time: 0.86 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2755934, upper bound: 4552.2755934
time: 1.04 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -5188.6577148, 4191.8315430, -3111.8627930, 2531.4621582, -7720.1201172, 7303.6943359
1: -4159.4165039, 4035.3022461, -2493.2846680, 2438.7373047, -6598.1538086, 6528.5869141
2: -5950.7929688, 4386.3300781, -3566.7409668, 2651.7770996, -8602.5683594, 7953.0708008
3: -2313.8149414, 5892.1474609, -1396.7198486, 3536.2380371, -5850.0527344, 7288.8662109
4: -6623.0805664, 4320.7236328, -3967.7014160, 2606.9890137, -9230.0693359, 8288.4248047

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

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
time: 0.92 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.61 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.61
Output dim: 0, lower bound: -4552.2755934, upper bound: 4552.2755934
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.61
Output dim: 0, lower bound: -4552.2755934, upper bound: 4552.2755934
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.61
Output dim: 0, lower bound: -4552.2755934, upper bound: 4552.2756825
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.61
Output dim: 0, lower bound: -4552.2755934, upper bound: 4552.2756825

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -2987.9899902, 2430.9243164, -2987.9899902, 2430.9243164, -5418.9140625, 5418.9140625
1: -2393.8061523, 2341.6025391, -2393.8061523, 2341.6025391, -4735.4086914, 4735.4086914
2: -3424.2626953, 2546.4648438, -3424.2626953, 2546.4648438, -5970.7275391, 5970.7275391
3: -1341.7878418, 3393.8977051, -1341.7878418, 3393.8977051, -4735.6855469, 4735.6855469
4: -3810.2302246, 2503.8750000, -3810.2302246, 2503.8750000, -6314.1054688, 6314.1054688

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2085840, upper bound: 4552.1950579
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1880421
time: 0.83 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -2987.9899902, 2430.9243164, -5188.6577148, 4191.8315430, -7179.8212891, 7619.5820312
1: -2393.8061523, 2341.6025391, -4159.4165039, 4035.3022461, -6429.1083984, 6501.0190430
2: -3424.2626953, 2546.4648438, -5950.7929688, 4386.3300781, -7810.5922852, 8497.2558594
3: -1341.7878418, 3393.8977051, -2313.8149414, 5892.1474609, -7233.9355469, 5707.7128906
4: -3810.2302246, 2503.8750000, -6623.0805664, 4320.7236328, -8130.9526367, 9126.9550781

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2085840, upper bound: 4552.1950579
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1880421
time: 1.11 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -5188.6577148, 4191.8315430, -2987.9899902, 2430.9243164, -7619.5820312, 7179.8212891
1: -4159.4165039, 4035.3022461, -2393.8061523, 2341.6025391, -6501.0190430, 6429.1083984
2: -5950.7929688, 4386.3300781, -3424.2626953, 2546.4648438, -8497.2558594, 7810.5922852
3: -2313.8149414, 5892.1474609, -1341.7878418, 3393.8977051, -5707.7128906, 7233.9355469
4: -6623.0805664, 4320.7236328, -3810.2302246, 2503.8750000, -9126.9550781, 8130.9531250

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1942569
time: 0.80 seconds

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

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1942569
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1905380
time: 0.87 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.47 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 0, lower bound: -4552.2085840, upper bound: 4552.1950579
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1880421
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 0, lower bound: -4552.2085840, upper bound: 4552.1950579
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1880421
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1942569
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1905380
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1942569
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1905380

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -2987.9899902, 2430.9243164, -5383.6064453, 5390.7915039
1: -2365.3845215, 2314.5366211, -2393.8061523, 2341.6025391, -4706.9873047, 4708.3427734
2: -3383.4587402, 2517.0407715, -3424.2626953, 2546.4648438, -5929.9228516, 5941.3027344
3: -1326.4201660, 3353.4309082, -1341.7878418, 3393.8977051, -4720.3173828, 4695.2187500
4: -3765.1281738, 2475.0305176, -3810.2302246, 2503.8750000, -6269.0029297, 6285.2602539

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1859666, upper bound: 4552.1859666
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1859666, upper bound: 4552.1880421
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4202.7128906, 3377.8999023, -2976.7226562, 2421.7700195, -6624.4829102, 6354.6210938
1: -3380.1076660, 3251.1071777, -2384.7473145, 2332.7238770, -5712.8315430, 5635.8540039
2: -4847.5078125, 3524.5412598, -3411.3022461, 2536.8029785, -7384.3105469, 6935.8432617
3: -1855.0902100, 4787.1733398, -1336.7360840, 3381.0124512, -5236.1025391, 6123.9086914
4: -5371.4536133, 3469.7426758, -3795.8500977, 2494.4553223, -7865.9091797, 7265.5922852

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1859666
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1880421
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -5188.6577148, 4191.8315430, -7144.5136719, 7591.4589844
1: -2365.3845215, 2314.5366211, -4159.4165039, 4035.3022461, -6400.6865234, 6473.9531250
2: -3383.4587402, 2517.0407715, -5950.7929688, 4386.3300781, -7769.7875977, 8467.8320312
3: -1326.4201660, 3353.4309082, -2313.8149414, 5892.1474609, -7218.5668945, 5667.2460938
4: -3765.1281738, 2475.0305176, -6623.0805664, 4320.7236328, -8085.8510742, 9098.1093750

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1884626, upper bound: 4552.1856700
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1884626, upper bound: 4552.1880421
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4200.2275391, 3376.0102539, -5175.4057617, 4181.1166992, -8381.3437500, 8551.4160156
1: -3378.0576172, 3249.2768555, -4148.7426758, 4024.9902344, -7403.0478516, 7398.0190430
2: -4844.5336914, 3522.5927734, -5935.5375977, 4375.1381836, -9219.6699219, 9458.1308594
3: -1854.0957031, 4784.2739258, -2307.9543457, 5877.0166016, -7731.1123047, 7092.2285156
4: -5368.2031250, 3467.8513184, -6606.1806641, 4309.7270508, -9677.9296875, 10074.0312500

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905380, upper bound: 4552.1856700
time: 1.61 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905380, upper bound: 4552.1880421
time: 1.71 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -2987.9899902, 2430.9243164, -7589.3227539, 7155.5288086
1: -4135.0488281, 4011.9240723, -2393.8061523, 2341.6025391, -6476.6513672, 6405.7290039
2: -5915.9174805, 4360.9184570, -3424.2626953, 2546.4648438, -8462.3818359, 7785.1806641
3: -2300.4028320, 5857.5903320, -1341.7878418, 3393.8977051, -5694.3007812, 7199.3779297
4: -6584.4262695, 4295.7910156, -3810.2302246, 2503.8750000, -9088.3007812, 8106.0195312

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1884626
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1905380
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6427.7465820, 5168.6801758, -2976.7226562, 2421.7700195, -8849.5146484, 8145.4013672
1: -5157.6206055, 4970.1308594, -2384.7473145, 2332.7238770, -7490.3447266, 7354.8779297
2: -7386.5043945, 5401.5595703, -3411.3022461, 2536.8029785, -9923.3076172, 8812.8593750
3: -2849.8481445, 7301.7617188, -1336.7360840, 3381.0124512, -6230.8603516, 8638.4980469
4: -8201.5048828, 5324.3559570, -3795.8500977, 2494.4553223, -10695.9589844, 9120.2060547

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1884626
time: 0.93 seconds

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

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1881660
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1905380
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6425.8520508, 5167.2119141, -5175.4057617, 4181.1166992, -10606.9687500, 10338.9990234
1: -5156.0722656, 4968.7270508, -4148.7426758, 4024.9902344, -9181.0615234, 9116.1416016
2: -7384.2504883, 5400.0463867, -5935.5375977, 4375.1381836, -11751.1933594, 11326.3291016
3: -2849.0493164, 7299.5883789, -2307.9543457, 5877.0166016, -8722.6455078, 9587.4726562
4: -8199.0634766, 5322.8666992, -6606.1806641, 4309.7270508, -12503.6699219, 11916.4023438

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1881660
time: 1.25 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1905380
time: 1.06 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.02 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -4552.1859666, upper bound: 4552.1859666
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -4552.1859666, upper bound: 4552.1880421
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1859666
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1880421
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -4552.1884626, upper bound: 4552.1856700
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -4552.1884626, upper bound: 4552.1880421
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -4552.1905380, upper bound: 4552.1856700
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -4552.1905380, upper bound: 4552.1880421
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1884626
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1905380
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1884626
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1905380
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1881660
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1905380
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1881660
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1905380

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -2952.6821289, 2402.8015137, -5355.4833984, 5355.4833984
1: -2365.3845215, 2314.5366211, -2365.3845215, 2314.5366211, -4679.9208984, 4679.9208984
2: -3383.4587402, 2517.0407715, -3383.4587402, 2517.0407715, -5900.4965820, 5900.4970703
3: -1326.4201660, 3353.4309082, -1326.4201660, 3353.4309082, -4679.8505859, 4679.8505859
4: -3765.1281738, 2475.0305176, -3765.1281738, 2475.0305176, -6240.1582031, 6240.1582031

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6986709, upper bound: 4551.7637953
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7920228, upper bound: 4551.7656775
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -4185.1938477, 3364.5900879, -6317.2719727, 6587.9951172
1: -2365.3845215, 2314.5366211, -3365.6628418, 3238.2065430, -5603.5908203, 5680.1992188
2: -3383.4587402, 2517.0407715, -4826.5581055, 3510.8107910, -6894.2695312, 7343.5976562
3: -1326.4201660, 3353.4309082, -1848.0799561, 4766.7568359, -6093.1762695, 5201.5102539
4: -3765.1281738, 2475.0305176, -5348.5498047, 3456.4106445, -7221.5380859, 7823.5800781

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6986709, upper bound: 4551.7637953
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7920228, upper bound: 4551.7656775
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4185.1938477, 3364.5900879, -2952.6821289, 2402.8015137, -6587.9951172, 6317.2719727
1: -3365.6628418, 3238.2065430, -2365.3845215, 2314.5366211, -5680.1992188, 5603.5908203
2: -4826.5581055, 3510.8107910, -3383.4587402, 2517.0407715, -7343.5971680, 6894.2695312
3: -1848.0799561, 4766.7568359, -1326.4201660, 3353.4309082, -5201.5102539, 6093.1767578
4: -5348.5498047, 3456.4106445, -3765.1281738, 2475.0305176, -7823.5800781, 7221.5380859

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7282293, upper bound: 4551.7583069
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7536229, upper bound: 4551.7536229
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4267.7421875, 3427.3356934, -4267.7421875, 3427.3356934, -7695.0771484, 7695.0771484
1: -3433.8293457, 3298.9086914, -3433.8293457, 3298.9086914, -6732.7377930, 6732.7377930
2: -4925.3828125, 3575.4016113, -4925.3828125, 3575.4016113, -8500.7832031, 8500.7841797
3: -1881.0106201, 4863.0942383, -1881.0106201, 4863.0942383, -6744.1049805, 6744.1049805
4: -5456.5566406, 3519.1264648, -5456.5566406, 3519.1264648, -8975.6835938, 8975.6835938

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7282293, upper bound: 4551.7583069
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7536229, upper bound: 4551.7536229
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -5158.3984375, 4167.5395508, -7120.2216797, 7561.2001953
1: -2365.3845215, 2314.5366211, -4135.0488281, 4011.9240723, -6377.3085938, 6449.5854492
2: -3383.4587402, 2517.0407715, -5915.9174805, 4360.9184570, -7744.3754883, 8432.9580078
3: -1326.4201660, 3353.4309082, -2300.4028320, 5857.5903320, -7184.0102539, 5653.8339844
4: -3765.1281738, 2475.0305176, -6584.4262695, 4295.7910156, -8060.9179688, 9059.4570312

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7052373, upper bound: 4551.7637288
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7985891, upper bound: 4551.7656122
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -6414.4458008, 5158.3549805, -8111.0371094, 8817.2470703
1: -2365.3845215, 2314.5366211, -5146.8022461, 4960.2553711, -7325.6396484, 7461.3388672
2: -3383.4587402, 2517.0407715, -7370.6708984, 5390.9257812, -8774.3847656, 9887.7119141
3: -1326.4201660, 3353.4309082, -2844.2338867, 7286.4814453, -8612.9013672, 6197.6650391
4: -3765.1281738, 2475.0305176, -8184.3623047, 5313.8818359, -9079.0097656, 10659.3925781

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7052373, upper bound: 4551.7637684
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7985891, upper bound: 4551.7656494
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4182.2895508, 3362.3830566, -5158.3984375, 4167.5395508, -8349.8291016, 8520.7812500
1: -3363.2678223, 3236.0671387, -4135.0488281, 4011.9240723, -7375.1909180, 7371.1162109
2: -4823.0864258, 3508.5339355, -5915.9174805, 4360.9184570, -9184.0048828, 9424.4511719
3: -1846.9169922, 4763.3745117, -2300.4028320, 5857.5903320, -7704.5073242, 7063.7773438
4: -5344.7543945, 3454.1999512, -6584.4262695, 4295.7910156, -9640.5439453, 10038.6259766

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7347956, upper bound: 4551.7582405
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7601892, upper bound: 4551.7535565
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4267.7421875, 3427.3356934, -6465.6679688, 5197.9145508, -9465.6562500, 9893.0039062
1: -3433.8293457, 3298.9086914, -5188.6606445, 4998.1103516, -8431.9394531, 8487.5683594
2: -4925.3828125, 3575.4016113, -7431.6689453, 5431.6601562, -10357.0410156, 11007.0703125
3: -1881.0106201, 4863.0942383, -2865.7429199, 7345.2402344, -9226.2509766, 7724.6606445
4: -5456.5566406, 3519.1264648, -8250.3779297, 5354.0131836, -10810.5673828, 11769.5009766

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7347956, upper bound: 4551.7582713
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7601892, upper bound: 4551.7535565
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -2952.6821289, 2402.8015137, -7561.2001953, 7120.2216797
1: -4135.0488281, 4011.9240723, -2365.3845215, 2314.5366211, -6449.5854492, 6377.3085938
2: -5915.9174805, 4360.9184570, -3383.4587402, 2517.0407715, -8432.9580078, 7744.3750000
3: -2300.4028320, 5857.5903320, -1326.4201660, 3353.4309082, -5653.8339844, 7184.0102539
4: -6584.4262695, 4295.7910156, -3765.1281738, 2475.0305176, -9059.4570312, 8060.9179688

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1917854
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1871620
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -4182.2895508, 3362.3830566, -8520.7812500, 8349.8281250
1: -4135.0488281, 4011.9240723, -3363.2678223, 3236.0671387, -7371.1157227, 7375.1914062
2: -5915.9174805, 4360.9184570, -4823.0864258, 3508.5339355, -9424.4511719, 9184.0048828
3: -2300.4028320, 5857.5903320, -1846.9169922, 4763.3745117, -7063.7773438, 7704.5073242
4: -6584.4262695, 4295.7910156, -5344.7543945, 3454.1999512, -10038.6259766, 9640.5439453

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1938141
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1892375
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6414.4453125, 5158.3549805, -2952.6821289, 2402.8015137, -8817.2470703, 8111.0371094
1: -5146.8022461, 4960.2563477, -2365.3845215, 2314.5366211, -7461.3388672, 7325.6406250
2: -7370.6704102, 5390.9252930, -3383.4587402, 2517.0407715, -9887.7109375, 8774.3837891
3: -2844.2338867, 7286.4809570, -1326.4201660, 3353.4309082, -6197.6650391, 8612.9013672
4: -8184.3613281, 5313.8818359, -3765.1281738, 2475.0305176, -10659.3916016, 9079.0097656

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4646323, upper bound: 4552.1884538
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844829, upper bound: 4552.1861417
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6465.6679688, 5197.9145508, -4267.7421875, 3427.3356934, -9893.0039062, 9465.6562500
1: -5188.6606445, 4998.1103516, -3433.8293457, 3298.9086914, -8487.5683594, 8431.9394531
2: -7431.6689453, 5431.6601562, -4925.3828125, 3575.4016113, -11007.0703125, 10357.0410156
3: -2865.7429199, 7345.2402344, -1881.0106201, 4863.0942383, -7724.6611328, 9226.2509766
4: -8250.3779297, 5354.0131836, -5456.5566406, 3519.1264648, -11769.5009766, 10810.5673828

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1898073
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1872713
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -5158.3984375, 4167.5395508, -9325.9375000, 9325.9375000
1: -4135.0488281, 4011.9240723, -4135.0488281, 4011.9240723, -8146.9726562, 8146.9726562
2: -5915.9174805, 4360.9184570, -5915.9174805, 4360.9184570, -10274.1767578, 10274.1767578
3: -2300.4028320, 5857.5903320, -2300.4028320, 5857.5903320, -8157.9863281, 8157.9863281
4: -6584.4262695, 4295.7910156, -6584.4262695, 4295.7910156, -10875.1572266, 10875.1562500

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1916102
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1868654
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -6412.2436523, 5156.6450195, -10311.4404297, 10579.7832031
1: -4135.0488281, 4011.9240723, -5145.0195312, 4958.6206055, -9092.3554688, 9156.9433594
2: -5915.9174805, 4360.9184570, -7368.0488281, 5389.1655273, -11295.8056641, 11721.0742188
3: -2300.4028320, 5857.5903320, -2843.3049316, 7283.9501953, -9564.5488281, 8697.4072266
4: -6584.4262695, 4295.7910156, -8181.5249023, 5312.1459961, -11883.9414062, 12472.2968750

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1938141
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1892375
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6412.2436523, 5156.6450195, -5158.3984375, 4167.5395508, -10579.7832031, 10311.4404297
1: -5145.0195312, 4958.6206055, -4135.0488281, 4011.9240723, -9156.9433594, 9092.3554688
2: -7368.0488281, 5389.1655273, -5915.9174805, 4360.9184570, -11721.0742188, 11295.8056641
3: -2843.3049316, 7283.9501953, -2300.4028320, 5857.5903320, -8697.4072266, 9564.5478516
4: -8181.5249023, 5312.1459961, -6584.4262695, 4295.7910156, -12472.2968750, 11883.9404297

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4646323, upper bound: 4552.1819100
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844829, upper bound: 4552.1795979
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6465.6679688, 5197.9145508, -6465.6679688, 5197.9145508, -11659.2441406, 11659.2441406
1: -5188.6606445, 4998.1103516, -5188.6606445, 4998.1103516, -10185.4218750, 10185.4208984
2: -7431.6689453, 5431.6601562, -7431.6689453, 5431.6601562, -12847.5341797, 12847.5341797
3: -2865.7429199, 7345.2402344, -2865.7429199, 7345.2402344, -10186.5107422, 10186.5107422
4: -8250.3779297, 5354.0131836, -8250.3779297, 5354.0131836, -13591.3330078, 13591.3330078

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1898002
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1872592
time: 0.89 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.20 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.20
Output dim: 0, lower bound: -4551.6986709, upper bound: 4551.7637953
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.20
Output dim: 0, lower bound: -4551.7920228, upper bound: 4551.7656775
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.20
Output dim: 0, lower bound: -4551.6986709, upper bound: 4551.7637953
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.20
Output dim: 0, lower bound: -4551.7920228, upper bound: 4551.7656775
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.20
Output dim: 0, lower bound: -4551.7282293, upper bound: 4551.7583069
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.20
Output dim: 0, lower bound: -4551.7536229, upper bound: 4551.7536229
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.20
Output dim: 0, lower bound: -4551.7282293, upper bound: 4551.7583069
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.20
Output dim: 0, lower bound: -4551.7536229, upper bound: 4551.7536229
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.20
Output dim: 0, lower bound: -4551.7052373, upper bound: 4551.7637288
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.20
Output dim: 0, lower bound: -4551.7985891, upper bound: 4551.7656122
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.20
Output dim: 0, lower bound: -4551.7052373, upper bound: 4551.7637684
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.20
Output dim: 0, lower bound: -4551.7985891, upper bound: 4551.7656494
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.20
Output dim: 0, lower bound: -4551.7347956, upper bound: 4551.7582405
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.20
Output dim: 0, lower bound: -4551.7601892, upper bound: 4551.7535565
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.20
Output dim: 0, lower bound: -4551.7347956, upper bound: 4551.7582713
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.20
Output dim: 0, lower bound: -4551.7601892, upper bound: 4551.7535565
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1917854
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1871620
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1938141
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1892375
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 0, lower bound: -4551.4646323, upper bound: 4552.1884538
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 0, lower bound: -4552.1844829, upper bound: 4552.1861417
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1898073
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1872713
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1916102
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1868654
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1938141
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1892375
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 0, lower bound: -4551.4646323, upper bound: 4552.1819100
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 0, lower bound: -4552.1844829, upper bound: 4552.1795979
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1898002
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1872592

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -2952.6821289, 2402.8015137, -6999.9189453, 6660.2255859
1: -3687.0437012, 3569.1484375, -2365.3845215, 2314.5366211, -6001.5800781, 5934.5332031
2: -5276.2031250, 3879.7956543, -3383.4587402, 2517.0407715, -7793.2426758, 7263.2524414
3: -2047.8087158, 5221.5805664, -1326.4201660, 3353.4309082, -5401.2392578, 6547.9995117
4: -5870.9716797, 3822.0715332, -3765.1281738, 2475.0305176, -8346.0019531, 7587.1992188

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7377764, upper bound: 4551.7089559
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7400887, upper bound: 4551.8030399
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -2952.6821289, 2402.8015137, -7506.7626953, 7075.9648438
1: -4091.2307129, 3969.2382812, -2365.3845215, 2314.5366211, -6405.7670898, 6334.6230469
2: -5853.0830078, 4314.3935547, -3383.4587402, 2517.0407715, -8370.1240234, 7697.8510742
3: -2275.4304199, 5795.1562500, -1326.4201660, 3353.4309082, -5628.8608398, 7121.5761719
4: -6514.6850586, 4250.0151367, -3765.1281738, 2475.0305176, -8989.7158203, 8015.1425781

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7676629, upper bound: 4551.7029061
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7699334, upper bound: 4551.7962579
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -4171.9448242, 3354.5097656, -7951.6274414, 7879.4882812
1: -3687.0437012, 3569.1484375, -3354.7460938, 3228.4331055, -6915.4755859, 6923.8945312
2: -5276.2031250, 3879.7956543, -4810.7221680, 3500.4116211, -8776.6152344, 8690.5175781
3: -2047.8087158, 5221.5805664, -1842.7706299, 4751.3149414, -6799.1230469, 7064.3505859
4: -5870.9716797, 3822.0715332, -5331.2343750, 3446.3107910, -9317.2802734, 9153.3056641

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7322538, upper bound: 4551.7384968
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7280324, upper bound: 4551.7641719
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -4171.5146484, 3354.1826172, -8458.1425781, 8294.7968750
1: -4091.2307129, 3969.2382812, -3354.3923340, 3228.1157227, -7319.3457031, 7323.6308594
2: -5853.0830078, 4314.3935547, -4810.2084961, 3500.0737305, -9353.1552734, 9124.6015625
3: -2275.4304199, 5795.1562500, -1842.5982666, 4750.8139648, -7026.2436523, 7637.7534180
4: -6514.6850586, 4250.0151367, -5330.6718750, 3445.9829102, -9960.6679688, 9580.6875000

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7621116, upper bound: 4551.7324642
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7578902, upper bound: 4551.7578580
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6314.3940430, 5079.3979492, -2952.6821289, 2402.8015137, -8717.1953125, 8032.0800781
1: -5068.0664062, 4885.0107422, -2365.3845215, 2314.5366211, -7382.6030273, 7250.3945312
2: -7260.5356445, 5307.1269531, -3383.4587402, 2517.0407715, -9777.5761719, 8690.5849609
3: -2799.3662109, 7180.2177734, -1326.4201660, 3353.4309082, -6152.7968750, 8506.6376953
4: -8059.7324219, 5230.5312500, -3765.1281738, 2475.0305176, -10534.7607422, 8995.6572266

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4292608, upper bound: 4551.6875449
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4341009, upper bound: 4551.7947080
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6338.7446289, 5097.5942383, -2952.6821289, 2402.8015137, -8741.5458984, 8050.2763672
1: -5085.8378906, 4901.6201172, -2365.3845215, 2314.5366211, -7400.3745117, 7267.0048828
2: -7283.4956055, 5326.7832031, -3383.4587402, 2517.0407715, -9800.5361328, 8710.2402344
3: -2810.2521973, 7201.1977539, -1326.4201660, 3353.4309082, -6163.6831055, 8527.6181641
4: -8087.9404297, 5250.9960938, -3765.1281738, 2475.0305176, -10562.9707031, 9016.1240234

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7426399, upper bound: 4551.6863623
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7473527, upper bound: 4551.7935255
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6366.1660156, 5119.2890625, -4267.7421875, 3427.3356934, -9793.5019531, 9387.0302734
1: -5110.7729492, 4923.2451172, -3433.8293457, 3298.9086914, -8409.6816406, 8357.0742188
2: -7322.4482422, 5348.0849609, -4925.3828125, 3575.4016113, -10897.8486328, 10273.4677734
3: -2820.9536133, 7239.8808594, -1881.0106201, 4863.0942383, -7676.4799805, 9120.8203125
4: -8126.7597656, 5270.9204102, -5456.5566406, 3519.1264648, -11645.8867188, 10727.4765625

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4241378, upper bound: 4551.7171103
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4219976, upper bound: 4551.7437677
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6390.3413086, 5137.2963867, -4267.7421875, 3427.3356934, -9817.6767578, 9405.0390625
1: -5128.1879883, 4939.5961914, -3433.8293457, 3298.9086914, -8427.0957031, 8373.4257812
2: -7344.9951172, 5367.6391602, -4925.3828125, 3575.4016113, -10920.3964844, 10293.0214844
3: -2831.8232422, 7260.3583984, -1881.0106201, 4863.0942383, -7687.5229492, 9141.3691406
4: -8154.4711914, 5291.2568359, -5456.5566406, 3519.1264648, -11673.5957031, 10747.8134766

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7376181, upper bound: 4551.7160190
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7354778, upper bound: 4551.7425851
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -5158.3984375, 4167.5395508, -8764.6572266, 8865.9423828
1: -3687.0437012, 3569.1484375, -4135.0488281, 4011.9240723, -7698.9667969, 7704.1972656
2: -5276.2031250, 3879.7956543, -5915.9174805, 4360.9184570, -9632.9892578, 9791.7519531
3: -2047.8087158, 5221.5805664, -2300.4028320, 5857.5903320, -7904.4311523, 7519.7099609
4: -5870.9716797, 3822.0715332, -6584.4262695, 4295.7910156, -10160.3740234, 10400.6279297

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1621650, upper bound: 4552.1621650
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1621650, upper bound: 4552.1916969
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -5158.3984375, 4167.5395508, -9271.5009766, 9281.6806641
1: -4091.2307129, 3969.2382812, -4135.0488281, 4011.9240723, -8103.1542969, 8104.2871094
2: -5853.0830078, 4314.3935547, -5915.9174805, 4360.9184570, -10211.1474609, 10227.9863281
3: -2275.4304199, 5795.1562500, -2300.4028320, 5857.5903320, -8133.0200195, 8094.7338867
4: -6514.6850586, 4250.0151367, -6584.4262695, 4295.7910156, -10805.3154297, 10829.7656250

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1916969, upper bound: 4552.1621650
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1916969, upper bound: 4552.1916969
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -6404.3627930, 5150.5229492, -9742.4648438, 10111.9062500
1: -3687.0437012, 3569.1484375, -5138.6459961, 4952.7646484, -8637.5996094, 8707.7939453
2: -5276.2031250, 3879.7956543, -7358.6723633, 5382.8603516, -10648.3251953, 11229.4414062
3: -2047.8087158, 5221.5805664, -2839.9770508, 7274.8964844, -9302.1035156, 8055.8129883
4: -5870.9716797, 3822.0715332, -8171.3715820, 5305.9340820, -11162.9541016, 11987.6718750

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1545275, upper bound: 4551.4726641
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1513498, upper bound: 4552.1927693
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -6404.0395508, 5150.2700195, -10250.5527344, 10527.3212891
1: -4091.2307129, 3969.2382812, -5138.3847656, 4952.5224609, -9042.3525391, 9107.6230469
2: -5853.0830078, 4314.3935547, -7358.2871094, 5382.5986328, -11226.2226562, 11665.2978516
3: -2275.4304199, 5795.1562500, -2839.8388672, 7274.5234375, -9530.5126953, 8630.6982422
4: -6514.6850586, 4250.0151367, -8170.9526367, 5305.6777344, -11807.6396484, 12416.3945312

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1794140, upper bound: 4551.4667226
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1762363, upper bound: 4552.1865733
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6312.1918945, 5077.7006836, -5158.3984375, 4167.5395508, -10479.4990234, 10229.6982422
1: -5066.2504883, 4883.3857422, -4135.0488281, 4011.9240723, -9078.1748047, 9014.3251953
2: -7257.9072266, 5305.3847656, -5915.9174805, 4360.9184570, -11607.7988281, 11208.3427734
3: -2798.4475098, 7177.6875000, -2300.4028320, 5857.5903320, -8649.2031250, 9457.7607422
4: -8056.8876953, 5228.8120117, -6584.4262695, 4295.7910156, -12345.0214844, 11797.1640625

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4726529, upper bound: 4552.1554065
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4667226, upper bound: 4552.1802930
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6336.5537109, 5095.8994141, -5158.3984375, 4167.5395508, -10504.0917969, 10247.6259766
1: -5084.0634766, 4899.9980469, -4135.0488281, 4011.9240723, -9095.9873047, 9031.0039062
2: -7280.8886719, 5325.0366211, -5915.9174805, 4360.9184570, -11634.7822266, 11228.0791016
3: -2809.3298340, 7198.6865234, -2300.4028320, 5857.5903320, -8660.2597656, 9479.2226562
4: -8085.1171875, 5249.2758789, -6584.4262695, 4295.7910156, -12376.9970703, 11817.3583984

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1927691, upper bound: 4552.1529031
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1865733, upper bound: 4552.1777897
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6366.1660156, 5119.2890625, -6465.6679688, 5197.9145508, -11557.8154297, 11577.7519531
1: -5110.7729492, 4923.2451172, -5188.6606445, 4998.1103516, -10105.2763672, 10107.7431641
2: -7322.4482422, 5348.0849609, -7431.6689453, 5431.6601562, -12735.1406250, 12760.1982422
3: -2820.9536133, 7239.8808594, -2865.7429199, 7345.2402344, -10138.3300781, 10080.5986328
4: -8126.7597656, 5270.9204102, -8250.3779297, 5354.0131836, -13465.0605469, 13504.7070312

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

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
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6390.3413086, 5137.2963867, -6465.6679688, 5197.9145508, -11584.8183594, 11595.4892578
1: -5128.1879883, 4939.5961914, -5188.6606445, 4998.1103516, -10125.4355469, 10124.1347656
2: -7344.9951172, 5367.6391602, -7431.6689453, 5431.6601562, -12761.6904297, 12779.8447266
3: -2831.8232422, 7260.3583984, -2865.7429199, 7345.2402344, -10149.3730469, 10101.5341797
4: -8154.4711914, 5291.2568359, -8250.3779297, 5354.0131836, -13496.5107422, 13524.7314453

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844829, upper bound: 4551.4668377
time: 0.96 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844829, upper bound: 4552.1872591
time: 0.85 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.58 seconds
IS_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 0, lower bound: -4551.7377764, upper bound: 4551.7089559
IS_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 0, lower bound: -4551.7400887, upper bound: 4551.8030399
IS_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 0, lower bound: -4551.7676629, upper bound: 4551.7029061
IS_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 0, lower bound: -4551.7699334, upper bound: 4551.7962579
IS_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 0, lower bound: -4551.7322538, upper bound: 4551.7384968
IS_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 0, lower bound: -4551.7280324, upper bound: 4551.7641719
IS_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 0, lower bound: -4551.7621116, upper bound: 4551.7324642
IS_A2_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 0, lower bound: -4551.7578902, upper bound: 4551.7578580
IS_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 0, lower bound: -4550.4292608, upper bound: 4551.6875449
IS_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 0, lower bound: -4550.4341009, upper bound: 4551.7947080
IS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 0, lower bound: -4551.7426399, upper bound: 4551.6863623
IS_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 0, lower bound: -4551.7473527, upper bound: 4551.7935255
IS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 0, lower bound: -4550.4241378, upper bound: 4551.7171103
IS_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 0, lower bound: -4550.4219976, upper bound: 4551.7437677
IS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 0, lower bound: -4551.7376181, upper bound: 4551.7160190
IS_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 0, lower bound: -4551.7354778, upper bound: 4551.7425851
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -4552.1621650, upper bound: 4552.1621650
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -4552.1621650, upper bound: 4552.1916969
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -4552.1916969, upper bound: 4552.1621650
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -4552.1916969, upper bound: 4552.1916969
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -4552.1545275, upper bound: 4551.4726641
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -4552.1513498, upper bound: 4552.1927693
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -4552.1794140, upper bound: 4551.4667226
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -4552.1762363, upper bound: 4552.1865733
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -4551.4726529, upper bound: 4552.1554065
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -4551.4667226, upper bound: 4552.1802930
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -4552.1927691, upper bound: 4552.1529031
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -4552.1865733, upper bound: 4552.1777897
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 0, lower bound: -4551.4646321, upper bound: 4551.4668377
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1872591
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -4552.1844829, upper bound: 4551.4668377
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -4552.1844829, upper bound: 4552.1872591

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -4597.1176758, 3707.5437012, -8304.6611328, 8304.6611328
1: -3687.0437012, 3569.1484375, -3687.0437012, 3569.1484375, -7256.1918945, 7256.1918945
2: -5276.2031250, 3879.7956543, -5276.2031250, 3879.7956543, -9150.5644531, 9150.5644531
3: -2047.8087158, 5221.5805664, -2047.8087158, 5221.5805664, -7266.1542969, 7266.1542969
4: -5870.9716797, 3822.0715332, -5870.9716797, 3822.0715332, -9685.8447266, 9685.8437500

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.7139066, upper bound: 4551.7342625
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7323025, upper bound: 4551.7393340
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -5103.9614258, 4123.2832031, -8720.4003906, 8811.5039062
1: -3687.0437012, 3569.1484375, -4091.2307129, 3969.2382812, -7656.2822266, 7660.3789062
2: -5276.2031250, 3879.7956543, -5853.0830078, 4314.3935547, -9586.7988281, 9728.7226562
3: -2047.8087158, 5221.5805664, -2275.4304199, 5795.1562500, -7841.1791992, 7494.9267578
4: -5870.9716797, 3822.0715332, -6514.6850586, 4250.0151367, -10114.9833984, 10330.7871094

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.7139066, upper bound: 4551.7641203
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7323025, upper bound: 4551.7686492
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -4597.1176758, 3707.5437012, -8811.5029297, 8720.4003906
1: -4091.2307129, 3969.2382812, -3687.0437012, 3569.1484375, -7660.3789062, 7656.2822266
2: -5853.0830078, 4314.3935547, -5276.2031250, 3879.7956543, -9728.7226562, 9586.7988281
3: -2275.4304199, 5795.1562500, -2047.8087158, 5221.5805664, -7494.9267578, 7841.1787109
4: -6514.6850586, 4250.0151367, -5870.9716797, 3822.0715332, -10330.7861328, 10114.9833984

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6117698, upper bound: 4551.7330142
time: 1.04 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7621604, upper bound: 4551.7323027
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -5103.9614258, 4123.2832031, -9227.2421875, 9227.2431641
1: -4091.2307129, 3969.2382812, -4091.2307129, 3969.2382812, -8060.4687500, 8060.4687500
2: -5853.0830078, 4314.3935547, -5853.0830078, 4314.3935547, -10164.9560547, 10164.9570312
3: -2275.4304199, 5795.1562500, -2275.4304199, 5795.1562500, -8069.9511719, 8069.9506836
4: -6514.6850586, 4250.0151367, -6514.6850586, 4250.0151367, -10759.9257812, 10759.9248047

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6117698, upper bound: 4551.7623046
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7621604, upper bound: 4551.7590118
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -6304.3085938, 5071.6264648, -9660.7812500, 10010.8750000
1: -3687.0437012, 3569.1484375, -5059.7998047, 4877.5727539, -8559.6123047, 8628.9472656
2: -5276.2031250, 3879.7956543, -7248.5024414, 5299.1445312, -10560.9375000, 11116.1435547
3: -2047.8087158, 5221.5805664, -2795.1542969, 7168.6303711, -9195.3164062, 8007.6499023
4: -5870.9716797, 3822.0715332, -8046.7084961, 5222.6582031, -11076.2490234, 11860.3710938

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4445115, upper bound: 4550.4219729
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4968185, upper bound: 4550.4290304
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -6328.7255859, 5089.8344727, -9678.7158203, 10036.2695312
1: -3687.0437012, 3569.1484375, -5077.7285156, 4894.1938477, -8576.3076172, 8646.8759766
2: -5276.2031250, 3879.7956543, -7271.5717773, 5318.7939453, -10580.6738281, 11143.2138672
3: -2047.8087158, 5221.5805664, -2806.0322266, 7189.7021484, -9216.8466797, 8018.7011719
4: -5870.9716797, 3822.0715332, -8075.0332031, 5243.1259766, -11096.4550781, 11892.4472656

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4444965, upper bound: 4551.7388089
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7091438, upper bound: 4551.7455781
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -6303.9809570, 5071.3740234, -10168.8701172, 10427.2617188
1: -4091.2307129, 3969.2382812, -5059.5312500, 4877.3315430, -8964.3662109, 9028.7695312
2: -5853.0830078, 4314.3935547, -7248.1108398, 5298.8857422, -11138.8398438, 11551.9931641
3: -2275.4304199, 5795.1562500, -2795.0170898, 7168.2524414, -9423.7207031, 8582.5380859
4: -6514.6850586, 4250.0151367, -8046.2846680, 5222.4033203, -11720.9375000, 12289.0888672

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.1823498, upper bound: 4550.4219544
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4984063, upper bound: 4550.4219562
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -6328.4003906, 5089.5830078, -10186.8056641, 10451.6796875
1: -4091.2307129, 3969.2382812, -5077.4638672, 4893.9526367, -8981.0634766, 9046.7021484
2: -5853.0830078, 4314.3935547, -7271.1840820, 5318.5332031, -11158.5732422, 11579.0673828
3: -2275.4304199, 5795.1562500, -2805.8947754, 7189.3276367, -9445.2548828, 8593.5888672
4: -6514.6850586, 4250.0151367, -8074.6147461, 5242.8696289, -11741.1435547, 12321.1689453

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.5933311, upper bound: 4551.7388911
time: 1.24 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7376065, upper bound: 4551.7391450
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6304.3085938, 5071.6264648, -4597.1176758, 3707.5437012, -10010.8759766, 9660.7812500
1: -5059.7998047, 4877.5727539, -3687.0437012, 3569.1484375, -8628.9472656, 8559.6123047
2: -7248.5024414, 5299.1445312, -5276.2031250, 3879.7956543, -11116.1435547, 10560.9375000
3: -2795.1542969, 7168.6303711, -2047.8087158, 5221.5805664, -8007.6499023, 9195.3173828
4: -8046.7084961, 5222.6582031, -5870.9716797, 3822.0715332, -11860.3710938, 11076.2490234

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.8813579, upper bound: 4551.7139337
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4294125, upper bound: 4551.7125020
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6303.9809570, 5071.3740234, -5103.9614258, 4123.2832031, -10427.2617188, 10168.8701172
1: -5059.5312500, 4877.3315430, -4091.2307129, 3969.2382812, -9028.7695312, 8964.3662109
2: -7248.1108398, 5298.8857422, -5853.0830078, 4314.3935547, -11551.9931641, 11138.8388672
3: -2795.0170898, 7168.2524414, -2275.4304199, 5795.1562500, -8582.5380859, 9423.7207031
4: -8046.2846680, 5222.4033203, -6514.6850586, 4250.0151367, -12289.0898438, 11720.9375000

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.8753052, upper bound: 4551.7423962
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4223163, upper bound: 4551.7409645
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6328.7255859, 5089.8344727, -4597.1176758, 3707.5437012, -10036.2695312, 9678.7148438
1: -5077.7285156, 4894.1938477, -3687.0437012, 3569.1484375, -8646.8759766, 8576.3076172
2: -7271.5717773, 5318.7939453, -5276.2031250, 3879.7956543, -11143.2138672, 10580.6738281
3: -2806.0322266, 7189.7021484, -2047.8087158, 5221.5805664, -8018.7011719, 9216.8476562
4: -8075.0332031, 5243.1259766, -5870.9716797, 3822.0715332, -11892.4472656, 11096.4550781

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7212073, upper bound: 4551.7124846
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7455781, upper bound: 4551.7124870
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6328.4003906, 5089.5830078, -5103.9614258, 4123.2832031, -10451.6796875, 10186.8066406
1: -5077.4638672, 4893.9526367, -4091.2307129, 3969.2382812, -9046.7021484, 8981.0634766
2: -7271.1840820, 5318.5332031, -5853.0830078, 4314.3935547, -11579.0673828, 11158.5732422
3: -2805.8947754, 7189.3276367, -2275.4304199, 5795.1562500, -8593.5888672, 9445.2558594
4: -8074.6147461, 5242.8696289, -6514.6850586, 4250.0151367, -12321.1689453, 11741.1425781

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.5970334, upper bound: 4551.7409471
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7391450, upper bound: 4551.7409495
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6366.1660156, 5119.2890625, -6390.3413086, 5137.2963867, -11494.0605469, 11503.3271484
1: -5110.7729492, 4923.2451172, -5128.1879883, 4939.5961914, -10043.9892578, 10047.7578125
2: -7322.4482422, 5348.0849609, -7344.9951172, 5367.6391602, -12667.4521484, 12674.3535156
3: -2820.9536133, 7239.8808594, -2831.8232422, 7260.3583984, -10053.3544922, 10043.4609375
4: -8126.7597656, 5270.9204102, -8154.4711914, 5291.2568359, -13398.4580078, 13409.8847656

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.7561070, upper bound: 4551.7420811
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4258744, upper bound: 4551.7417620
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6390.3413086, 5137.2963867, -6366.1660156, 5119.2890625, -11503.3271484, 11494.0615234
1: -5128.1879883, 4939.5961914, -5110.7729492, 4923.2451172, -10047.7568359, 10043.9892578
2: -7344.9951172, 5367.6391602, -7322.4482422, 5348.0849609, -12674.3544922, 12667.4521484
3: -2831.8232422, 7260.3583984, -2820.9536133, 7239.8808594, -10043.4609375, 10053.3535156
4: -8154.4711914, 5291.2568359, -8126.7597656, 5270.9204102, -13409.8847656, 13398.4580078

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.5549222, upper bound: 4550.4258699
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.5417017, upper bound: 4550.4273814
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6390.3413086, 5137.2963867, -6390.3413086, 5137.2963867, -11521.0634766, 11521.0644531
1: -5128.1879883, 4939.5961914, -5128.1879883, 4939.5961914, -10064.1494141, 10064.1494141
2: -7344.9951172, 5367.6391602, -7344.9951172, 5367.6391602, -12694.0019531, 12694.0019531
3: -2831.8232422, 7260.3583984, -2831.8232422, 7260.3583984, -10064.3964844, 10064.3964844
4: -8154.4711914, 5291.2568359, -8154.4711914, 5291.2568359, -13429.9082031, 13429.9091797

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.5549223, upper bound: 4551.7333355
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.5417017, upper bound: 4551.7375208
time: 0.78 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.42 seconds
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 0, lower bound: -4550.7139066, upper bound: 4551.7342625
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 0, lower bound: -4551.7323025, upper bound: 4551.7393340
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 0, lower bound: -4550.7139066, upper bound: 4551.7641203
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 0, lower bound: -4551.7323025, upper bound: 4551.7686492
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 0, lower bound: -4551.6117698, upper bound: 4551.7330142
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 0, lower bound: -4551.7621604, upper bound: 4551.7323027
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 0, lower bound: -4551.6117698, upper bound: 4551.7623046
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 0, lower bound: -4551.7621604, upper bound: 4551.7590118
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 0, lower bound: -4550.4445115, upper bound: 4550.4219729
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 0, lower bound: -4551.4968185, upper bound: 4550.4290304
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 0, lower bound: -4550.4444965, upper bound: 4551.7388089
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 0, lower bound: -4551.7091438, upper bound: 4551.7455781
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 0, lower bound: -4551.1823498, upper bound: 4550.4219544
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 0, lower bound: -4551.4984063, upper bound: 4550.4219562
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 0, lower bound: -4551.5933311, upper bound: 4551.7388911
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 0, lower bound: -4551.7376065, upper bound: 4551.7391450
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 0, lower bound: -4550.8813579, upper bound: 4551.7139337
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 0, lower bound: -4550.4294125, upper bound: 4551.7125020
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 0, lower bound: -4550.8753052, upper bound: 4551.7423962
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 0, lower bound: -4550.4223163, upper bound: 4551.7409645
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 0, lower bound: -4551.7212073, upper bound: 4551.7124846
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 0, lower bound: -4551.7455781, upper bound: 4551.7124870
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 0, lower bound: -4551.5970334, upper bound: 4551.7409471
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 0, lower bound: -4551.7391450, upper bound: 4551.7409495
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 0, lower bound: -4550.7561070, upper bound: 4551.7420811
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 0, lower bound: -4550.4258744, upper bound: 4551.7417620
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 0, lower bound: -4551.5549222, upper bound: 4550.4258699
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 0, lower bound: -4551.5417017, upper bound: 4550.4273814
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 0, lower bound: -4551.5549223, upper bound: 4551.7333355
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.42
Output dim: 0, lower bound: -4551.5417017, upper bound: 4551.7375208
Binary search (step 0): status=Status.VERIFIED, low=0.5000000, high=1.0000000, mid=0.5000000, abs_max=5687.5751953125
rel_dist={0: [-4552.2878636421, 4552.287863642101]}

## Binary search (step 1) starts
Candidate diff: 0.7500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2865166, upper bound: 4552.2783404
time: 0.69 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711
time: 1.77 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.61 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.61
Output dim: 0, lower bound: -4552.2865166, upper bound: 4552.2783404
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.61
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -2987.9899902, 2430.9243164, -3136.6748047, 2550.9006348, -5538.8906250, 5567.5991211
1: -2393.8061523, 2341.6025391, -2513.2941895, 2457.4460449, -4851.2519531, 4854.8964844
2: -3424.2626953, 2546.4648438, -3595.4755859, 2672.1398926, -6096.4023438, 6141.9404297
3: -1341.7878418, 3393.8977051, -1407.5072021, 3564.2558594, -4906.0439453, 4801.4047852
4: -3810.2302246, 2503.8750000, -3999.4067383, 2627.0041504, -6437.2343750, 6503.2817383

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711
time: 0.81 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711
time: 0.85 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -5188.6577148, 4191.8315430, -3134.5205078, 2549.2131348, -7737.8710938, 7326.3510742
1: -4159.4165039, 4035.3022461, -2511.5576172, 2455.8217773, -6615.2382812, 6546.8598633
2: -5950.7929688, 4386.3300781, -3592.9816895, 2670.3718262, -8621.1650391, 7979.3110352
3: -2313.8149414, 5892.1474609, -1406.5705566, 3561.8251953, -5875.6401367, 7298.7172852
4: -6623.0805664, 4320.7236328, -3996.6542969, 2625.2661133, -9248.3447266, 8317.3779297

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711
time: 0.87 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711
time: 0.76 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.31 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -2987.9899902, 2430.9243164, -2987.9899902, 2430.9243164, -5418.9140625, 5418.9140625
1: -2393.8061523, 2341.6025391, -2393.8061523, 2341.6025391, -4735.4086914, 4735.4086914
2: -3424.2626953, 2546.4648438, -3424.2626953, 2546.4648438, -5970.7275391, 5970.7275391
3: -1341.7878418, 3393.8977051, -1341.7878418, 3393.8977051, -4735.6855469, 4735.6855469
4: -3810.2302246, 2503.8750000, -3810.2302246, 2503.8750000, -6314.1054688, 6314.1054688

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2085840, upper bound: 4552.1951259
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1880421
time: 0.80 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -2987.9899902, 2430.9243164, -5188.6577148, 4191.8315430, -7179.8212891, 7619.5820312
1: -2393.8061523, 2341.6025391, -4159.4165039, 4035.3022461, -6429.1083984, 6501.0190430
2: -3424.2626953, 2546.4648438, -5950.7929688, 4386.3300781, -7810.5922852, 8497.2558594
3: -1341.7878418, 3393.8977051, -2313.8149414, 5892.1474609, -7233.9355469, 5707.7128906
4: -3810.2302246, 2503.8750000, -6623.0805664, 4320.7236328, -8130.9526367, 9126.9550781

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2085840, upper bound: 4552.1951259
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1880421
time: 0.91 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -5188.6577148, 4191.8315430, -2987.9899902, 2430.9243164, -7619.5820312, 7179.8212891
1: -4159.4165039, 4035.3022461, -2393.8061523, 2341.6025391, -6501.0190430, 6429.1083984
2: -5950.7929688, 4386.3300781, -3424.2626953, 2546.4648438, -8497.2558594, 7810.5922852
3: -2313.8149414, 5892.1474609, -1341.7878418, 3393.8977051, -5707.7128906, 7233.9355469
4: -6623.0805664, 4320.7236328, -3810.2302246, 2503.8750000, -9126.9550781, 8130.9531250

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1946649
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1905380
time: 0.86 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -5188.6577148, 4191.8315430, -5188.6577148, 4191.8315430, -9380.4892578, 9380.4892578
1: -4159.4165039, 4035.3022461, -4159.4165039, 4035.3022461, -8194.7187500, 8194.7187500
2: -5950.7929688, 4386.3300781, -5950.7929688, 4386.3300781, -10334.5771484, 10334.5771484
3: -2313.8149414, 5892.1474609, -2313.8149414, 5892.1474609, -8205.9628906, 8205.9628906
4: -6623.0805664, 4320.7236328, -6623.0805664, 4320.7236328, -10938.7910156, 10938.7919922

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1946649
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1905380
time: 0.96 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.78 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.78
Output dim: 0, lower bound: -4552.2085840, upper bound: 4552.1951259
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.78
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1880421
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.78
Output dim: 0, lower bound: -4552.2085840, upper bound: 4552.1951259
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.78
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1880421
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.78
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1946649
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.78
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1905380
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.78
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1946649
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.78
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1905380

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -2987.9899902, 2430.9243164, -5383.6064453, 5390.7915039
1: -2365.3845215, 2314.5366211, -2393.8061523, 2341.6025391, -4706.9873047, 4708.3427734
2: -3383.4587402, 2517.0407715, -3424.2626953, 2546.4648438, -5929.9228516, 5941.3027344
3: -1326.4201660, 3353.4309082, -1341.7878418, 3393.8977051, -4720.3173828, 4695.2187500
4: -3765.1281738, 2475.0305176, -3810.2302246, 2503.8750000, -6269.0029297, 6285.2602539

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1859666, upper bound: 4552.1859666
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1859666, upper bound: 4552.1880421
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4211.4248047, 3384.5349121, -2985.1865234, 2428.6462402, -6640.0703125, 6369.7216797
1: -3387.2963867, 3257.5280762, -2391.5520020, 2339.3930664, -5726.6894531, 5649.0791016
2: -4857.9379883, 3531.3728027, -3421.0366211, 2544.0610352, -7401.9990234, 6952.4091797
3: -1858.5767822, 4797.3505859, -1340.5312500, 3390.6906738, -5249.2675781, 6137.8813477
4: -5382.8540039, 3476.3754883, -3806.6516113, 2501.5310059, -7884.3847656, 7283.0268555

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1859666
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1880421
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -5188.6577148, 4191.8315430, -7144.5136719, 7591.4589844
1: -2365.3845215, 2314.5366211, -4159.4165039, 4035.3022461, -6400.6865234, 6473.9531250
2: -3383.4587402, 2517.0407715, -5950.7929688, 4386.3300781, -7769.7875977, 8467.8320312
3: -1326.4201660, 3353.4309082, -2313.8149414, 5892.1474609, -7218.5668945, 5667.2460938
4: -3765.1281738, 2475.0305176, -6623.0805664, 4320.7236328, -8085.8510742, 9098.1093750

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1884626, upper bound: 4552.1856700
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1884626, upper bound: 4552.1880421
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4209.1367188, 3382.7915039, -5185.4584961, 4189.2436523, -8398.3808594, 8568.2480469
1: -3385.4074707, 3255.8413086, -4156.8403320, 4032.8115234, -7418.2187500, 7412.6816406
2: -4855.1977539, 3529.5781250, -5947.1103516, 4383.6279297, -9238.8261719, 9476.6875000
3: -1857.6608887, 4794.6767578, -2312.3999023, 5888.4936523, -7746.1542969, 7107.0766602
4: -5379.8583984, 3474.6325684, -6619.0000000, 4318.0683594, -9697.9267578, 10093.6328125

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905380, upper bound: 4552.1856700
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905380, upper bound: 4552.1880421
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -2987.9899902, 2430.9243164, -7589.3227539, 7155.5288086
1: -4135.0488281, 4011.9240723, -2393.8061523, 2341.6025391, -6476.6513672, 6405.7290039
2: -5915.9174805, 4360.9184570, -3424.2626953, 2546.4648438, -8462.3818359, 7785.1806641
3: -2300.4028320, 5857.5903320, -1341.7878418, 3393.8977051, -5694.3007812, 7199.3779297
4: -6584.4262695, 4295.7910156, -3810.2302246, 2503.8750000, -9088.3007812, 8106.0195312

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1884626
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1905380
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6434.4213867, 5173.8354492, -2985.1865234, 2428.6462402, -8863.0664062, 8159.0200195
1: -5163.0742188, 4975.0620117, -2391.5520020, 2339.3930664, -7502.4672852, 7366.6132812
2: -7394.4443359, 5406.8676758, -3421.0366211, 2544.0610352, -9938.5058594, 8827.9042969
3: -2852.6525879, 7309.4101562, -1340.5312500, 3390.6906738, -6243.3432617, 8649.9404297
4: -8210.0996094, 5329.5864258, -3806.6516113, 2501.5310059, -10711.6298828, 9136.2382812

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1884626
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1905380
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -5188.6577148, 4191.8315430, -9350.2294922, 9356.1972656
1: -4135.0488281, 4011.9240723, -4159.4165039, 4035.3022461, -8170.3510742, 8171.3408203
2: -5915.9174805, 4360.9184570, -5950.7929688, 4386.3300781, -10299.6093750, 10309.1445312
3: -2300.4028320, 5857.5903320, -2313.8149414, 5892.1474609, -8192.5507812, 8171.4052734
4: -6584.4262695, 4295.7910156, -6623.0805664, 4320.7236328, -10900.1123047, 10913.8359375

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1881660
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1905380
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6432.6616211, 5172.4794922, -5185.4584961, 4189.2436523, -10621.9052734, 10354.3281250
1: -5161.6367188, 4973.7661133, -4156.8403320, 4032.8115234, -9194.4482422, 9129.2783203
2: -7392.3510742, 5405.4707031, -5947.1103516, 4383.6279297, -11767.6503906, 11343.3496094
3: -2851.9150391, 7307.3930664, -2312.3999023, 5888.4936523, -8737.0419922, 9599.5908203
4: -8207.8339844, 5328.2109375, -6619.0000000, 4318.0683594, -12520.7441406, 11934.5781250

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1881660
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1905380
time: 0.72 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.68 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -4552.1859666, upper bound: 4552.1859666
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -4552.1859666, upper bound: 4552.1880421
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1859666
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1880421
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -4552.1884626, upper bound: 4552.1856700
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -4552.1884626, upper bound: 4552.1880421
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -4552.1905380, upper bound: 4552.1856700
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -4552.1905380, upper bound: 4552.1880421
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1884626
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1905380
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1884626
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1905380
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1881660
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1905380
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1881660
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1905380

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -2952.6821289, 2402.8015137, -5355.4833984, 5355.4833984
1: -2365.3845215, 2314.5366211, -2365.3845215, 2314.5366211, -4679.9208984, 4679.9208984
2: -3383.4587402, 2517.0407715, -3383.4587402, 2517.0407715, -5900.4965820, 5900.4970703
3: -1326.4201660, 3353.4309082, -1326.4201660, 3353.4309082, -4679.8505859, 4679.8505859
4: -3765.1281738, 2475.0305176, -3765.1281738, 2475.0305176, -6240.1582031, 6240.1582031

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6986709, upper bound: 4551.7638377
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7920228, upper bound: 4551.7659535
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -4195.3359375, 3372.2932129, -6324.9755859, 6598.1376953
1: -2365.3845215, 2314.5366211, -3374.0241699, 3245.6728516, -5611.0566406, 5688.5605469
2: -3383.4587402, 2517.0407715, -4838.6835938, 3518.7585449, -6902.2158203, 7355.7231445
3: -1326.4201660, 3353.4309082, -1852.1385498, 4778.5722656, -6104.9916992, 5205.5693359
4: -3765.1281738, 2475.0305176, -5361.8071289, 3464.1286621, -7229.2548828, 7836.8378906

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6986709, upper bound: 4551.7638377
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7920228, upper bound: 4551.7659535
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4195.3359375, 3372.2932129, -2952.6821289, 2402.8015137, -6598.1376953, 6324.9755859
1: -3374.0241699, 3245.6728516, -2365.3845215, 2314.5366211, -5688.5605469, 5611.0566406
2: -4838.6835938, 3518.7585449, -3383.4587402, 2517.0407715, -7355.7231445, 6902.2163086
3: -1852.1385498, 4778.5722656, -1326.4201660, 3353.4309082, -5205.5693359, 6104.9921875
4: -5361.8071289, 3464.1286621, -3765.1281738, 2475.0305176, -7836.8378906, 7229.2548828

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7282293, upper bound: 4551.7583069
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7536229, upper bound: 4551.7536229
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4267.7421875, 3427.3356934, -4267.7421875, 3427.3356934, -7695.0771484, 7695.0771484
1: -3433.8293457, 3298.9086914, -3433.8293457, 3298.9086914, -6732.7377930, 6732.7377930
2: -4925.3828125, 3575.4016113, -4925.3828125, 3575.4016113, -8500.7832031, 8500.7841797
3: -1881.0106201, 4863.0942383, -1881.0106201, 4863.0942383, -6744.1049805, 6744.1049805
4: -5456.5566406, 3519.1264648, -5456.5566406, 3519.1264648, -8975.6835938, 8975.6835938

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7282293, upper bound: 4551.7583069
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7536229, upper bound: 4551.7536229
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -5158.3984375, 4167.5395508, -7120.2216797, 7561.2001953
1: -2365.3845215, 2314.5366211, -4135.0488281, 4011.9240723, -6377.3085938, 6449.5854492
2: -3383.4587402, 2517.0407715, -5915.9174805, 4360.9184570, -7744.3754883, 8432.9580078
3: -1326.4201660, 3353.4309082, -2300.4028320, 5857.5903320, -7184.0102539, 5653.8339844
4: -3765.1281738, 2475.0305176, -6584.4262695, 4295.7910156, -8060.9179688, 9059.4570312

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7052373, upper bound: 4551.7637713
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7985891, upper bound: 4551.7658871
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -6422.1362305, 5164.3261719, -8117.0083008, 8824.9365234
1: -2365.3845215, 2314.5366211, -5153.0366211, 4965.9672852, -7331.3515625, 7467.5732422
2: -3383.4587402, 2517.0407715, -7379.8271484, 5397.0751953, -8780.5312500, 9896.8652344
3: -1326.4201660, 3353.4309082, -2847.4799805, 7295.3183594, -8621.7382812, 6200.9111328
4: -3765.1281738, 2475.0305176, -8194.2763672, 5319.9399414, -9085.0673828, 10669.3037109

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7052373, upper bound: 4551.7638109
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7985891, upper bound: 4551.7659267
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4192.6772461, 3370.2736816, -5158.3984375, 4167.5395508, -8360.2167969, 8528.6718750
1: -3371.8327637, 3243.7153320, -4135.0488281, 4011.9240723, -7383.7568359, 7378.7636719
2: -4835.5048828, 3516.6752930, -5915.9174805, 4360.9184570, -9196.4218750, 9432.5927734
3: -1851.0748291, 4775.4750977, -2300.4028320, 5857.5903320, -7708.6650391, 7075.8779297
4: -5358.3320312, 3462.1057129, -6584.4262695, 4295.7910156, -9654.1201172, 10046.5312500

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7347954, upper bound: 4551.7582405
time: 1.22 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7601892, upper bound: 4551.7535565
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4267.7421875, 3427.3356934, -6465.6679688, 5197.9145508, -9465.6562500, 9893.0039062
1: -3433.8293457, 3298.9086914, -5188.6606445, 4998.1103516, -8431.9394531, 8487.5683594
2: -4925.3828125, 3575.4016113, -7431.6689453, 5431.6601562, -10357.0410156, 11007.0703125
3: -1881.0106201, 4863.0942383, -2865.7429199, 7345.2402344, -9226.2509766, 7724.6606445
4: -5456.5566406, 3519.1264648, -8250.3779297, 5354.0131836, -10810.5673828, 11769.5009766

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7347956, upper bound: 4551.7582713
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7601892, upper bound: 4551.7535565
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -2952.6821289, 2402.8015137, -7561.2001953, 7120.2216797
1: -4135.0488281, 4011.9240723, -2365.3845215, 2314.5366211, -6449.5854492, 6377.3085938
2: -5915.9174805, 4360.9184570, -3383.4587402, 2517.0407715, -8432.9580078, 7744.3750000
3: -2300.4028320, 5857.5903320, -1326.4201660, 3353.4309082, -5653.8339844, 7184.0102539
4: -6584.4262695, 4295.7910156, -3765.1281738, 2475.0305176, -9059.4570312, 8060.9179688

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1924999
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1871620
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -4192.6772461, 3370.2736816, -8528.6718750, 8360.2167969
1: -4135.0488281, 4011.9240723, -3371.8327637, 3243.7153320, -7378.7636719, 7383.7568359
2: -5915.9174805, 4360.9184570, -4835.5048828, 3516.6752930, -9432.5927734, 9196.4228516
3: -2300.4028320, 5857.5903320, -1851.0748291, 4775.4750977, -7075.8779297, 7708.6650391
4: -6584.4262695, 4295.7910156, -5358.3320312, 3462.1057129, -10046.5322266, 9654.1201172

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1945753
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1892375
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6422.1362305, 5164.3261719, -2952.6821289, 2402.8015137, -8824.9365234, 8117.0083008
1: -5153.0366211, 4965.9672852, -2365.3845215, 2314.5366211, -7467.5732422, 7331.3515625
2: -7379.8271484, 5397.0751953, -3383.4587402, 2517.0407715, -9896.8652344, 8780.5322266
3: -2847.4799805, 7295.3183594, -1326.4201660, 3353.4309082, -6200.9111328, 8621.7382812
4: -8194.2763672, 5319.9399414, -3765.1281738, 2475.0305176, -10669.3037109, 9085.0673828

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4646323, upper bound: 4552.1884538
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1861417
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6465.6679688, 5197.9145508, -4267.7421875, 3427.3356934, -9893.0039062, 9465.6562500
1: -5188.6606445, 4998.1103516, -3433.8293457, 3298.9086914, -8487.5683594, 8431.9394531
2: -7431.6689453, 5431.6601562, -4925.3828125, 3575.4016113, -11007.0703125, 10357.0410156
3: -2865.7429199, 7345.2402344, -1881.0106201, 4863.0942383, -7724.6611328, 9226.2509766
4: -8250.3779297, 5354.0131836, -5456.5566406, 3519.1264648, -11769.5009766, 10810.5673828

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1898073
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1872713
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -5158.3984375, 4167.5395508, -9325.9375000, 9325.9375000
1: -4135.0488281, 4011.9240723, -4135.0488281, 4011.9240723, -8146.9726562, 8146.9726562
2: -5915.9174805, 4360.9184570, -5915.9174805, 4360.9184570, -10274.1767578, 10274.1767578
3: -2300.4028320, 5857.5903320, -2300.4028320, 5857.5903320, -8157.9863281, 8157.9863281
4: -6584.4262695, 4295.7910156, -6584.4262695, 4295.7910156, -10875.1572266, 10875.1562500

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1922033
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1868654
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -6420.1196289, 5162.7602539, -10317.5478516, 10587.6591797
1: -4135.0488281, 4011.9240723, -5151.3920898, 4964.4687500, -9098.1943359, 9163.3164062
2: -5915.9174805, 4360.9184570, -7377.4257812, 5395.4628906, -11302.0888672, 11730.2753906
3: -2300.4028320, 5857.5903320, -2846.6281738, 7292.9985352, -9573.4345703, 8700.7207031
4: -6584.4262695, 4295.7910156, -8191.6752930, 5318.3500977, -11890.1376953, 12482.3896484

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1945753
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1892375
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6420.1196289, 5162.7602539, -5158.3984375, 4167.5395508, -10587.6591797, 10317.5478516
1: -5151.3920898, 4964.4687500, -4135.0488281, 4011.9240723, -9163.3164062, 9098.1943359
2: -7377.4257812, 5395.4628906, -5915.9174805, 4360.9184570, -11730.2753906, 11302.0888672
3: -2846.6281738, 7292.9985352, -2300.4028320, 5857.5903320, -8700.7216797, 9573.4345703
4: -8191.6752930, 5318.3500977, -6584.4262695, 4295.7910156, -12482.3906250, 11890.1376953

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4646323, upper bound: 4552.1819100
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844829, upper bound: 4552.1795979
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6465.6679688, 5197.9145508, -6465.6679688, 5197.9145508, -11659.2441406, 11659.2441406
1: -5188.6606445, 4998.1103516, -5188.6606445, 4998.1103516, -10185.4218750, 10185.4208984
2: -7431.6689453, 5431.6601562, -7431.6689453, 5431.6601562, -12847.5341797, 12847.5341797
3: -2865.7429199, 7345.2402344, -2865.7429199, 7345.2402344, -10186.5107422, 10186.5107422
4: -8250.3779297, 5354.0131836, -8250.3779297, 5354.0131836, -13591.3330078, 13591.3330078

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1898002
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1872592
time: 0.75 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.05 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.05
Output dim: 0, lower bound: -4551.6986709, upper bound: 4551.7638377
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.05
Output dim: 0, lower bound: -4551.7920228, upper bound: 4551.7659535
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.05
Output dim: 0, lower bound: -4551.6986709, upper bound: 4551.7638377
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 5.05
Output dim: 0, lower bound: -4551.7920228, upper bound: 4551.7659535
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.05
Output dim: 0, lower bound: -4551.7282293, upper bound: 4551.7583069
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.05
Output dim: 0, lower bound: -4551.7536229, upper bound: 4551.7536229
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.05
Output dim: 0, lower bound: -4551.7282293, upper bound: 4551.7583069
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 5.05
Output dim: 0, lower bound: -4551.7536229, upper bound: 4551.7536229
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.05
Output dim: 0, lower bound: -4551.7052373, upper bound: 4551.7637713
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.05
Output dim: 0, lower bound: -4551.7985891, upper bound: 4551.7658871
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.05
Output dim: 0, lower bound: -4551.7052373, upper bound: 4551.7638109
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 5.05
Output dim: 0, lower bound: -4551.7985891, upper bound: 4551.7659267
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.05
Output dim: 0, lower bound: -4551.7347954, upper bound: 4551.7582405
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.05
Output dim: 0, lower bound: -4551.7601892, upper bound: 4551.7535565
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.05
Output dim: 0, lower bound: -4551.7347956, upper bound: 4551.7582713
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 5.05
Output dim: 0, lower bound: -4551.7601892, upper bound: 4551.7535565
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.05
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1924999
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.05
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1871620
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.05
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1945753
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.05
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1892375
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.05
Output dim: 0, lower bound: -4551.4646323, upper bound: 4552.1884538
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.05
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1861417
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.05
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1898073
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.05
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1872713
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.05
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1922033
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.05
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1868654
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.05
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1945753
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.05
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1892375
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.05
Output dim: 0, lower bound: -4551.4646323, upper bound: 4552.1819100
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.05
Output dim: 0, lower bound: -4552.1844829, upper bound: 4552.1795979
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.05
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1898002
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.05
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1872592

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -2952.6821289, 2402.8015137, -6999.9189453, 6660.2255859
1: -3687.0437012, 3569.1484375, -2365.3845215, 2314.5366211, -6001.5800781, 5934.5332031
2: -5276.2031250, 3879.7956543, -3383.4587402, 2517.0407715, -7793.2426758, 7263.2524414
3: -2047.8087158, 5221.5805664, -1326.4201660, 3353.4309082, -5401.2392578, 6547.9995117
4: -5870.9716797, 3822.0715332, -3765.1281738, 2475.0305176, -8346.0019531, 7587.1992188

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7380623, upper bound: 4551.7103340
time: 1.69 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7401350, upper bound: 4551.8036859
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -2952.6821289, 2402.8015137, -7506.7626953, 7075.9648438
1: -4091.2307129, 3969.2382812, -2365.3845215, 2314.5366211, -6405.7670898, 6334.6230469
2: -5853.0830078, 4314.3935547, -3383.4587402, 2517.0407715, -8370.1240234, 7697.8510742
3: -2275.4304199, 5795.1562500, -1326.4201660, 3353.4309082, -5628.8608398, 7121.5761719
4: -6514.6850586, 4250.0151367, -3765.1281738, 2475.0305176, -8989.7158203, 8015.1425781

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7679201, upper bound: 4551.7029061
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7699927, upper bound: 4551.7962579
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -4183.1699219, 3363.0527344, -7960.1704102, 7890.7133789
1: -3687.0437012, 3569.1484375, -3363.9941406, 3236.7163086, -6923.7587891, 6933.1425781
2: -5276.2031250, 3879.7956543, -4824.1391602, 3509.2253418, -8785.4277344, 8703.9345703
3: -2047.8087158, 5221.5805664, -1847.2700195, 4764.4003906, -6812.2089844, 7068.8500977
4: -5870.9716797, 3822.0715332, -5345.9067383, 3454.8706055, -9325.8417969, 9167.9775391

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7322538, upper bound: 4551.7398922
time: 1.14 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7280324, upper bound: 4551.7652860
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -4182.7773438, 3362.7529297, -8466.7138672, 8306.0605469
1: -4091.2307129, 3969.2382812, -3363.6694336, 3236.4260254, -7327.6557617, 7332.9072266
2: -5853.0830078, 4314.3935547, -4823.6684570, 3508.9165039, -9361.9990234, 9138.0625000
3: -2275.4304199, 5795.1562500, -1847.1123047, 4763.9414062, -7039.3706055, 7642.2685547
4: -6514.6850586, 4250.0151367, -5345.3911133, 3454.5705566, -9969.2558594, 9595.4062500

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7621116, upper bound: 4551.7324642
time: 0.96 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7578902, upper bound: 4551.7578580
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6322.0839844, 5085.3212891, -2952.6821289, 2402.8015137, -8724.8857422, 8038.0034180
1: -5074.4047852, 4890.6782227, -2365.3845215, 2314.5366211, -7388.9414062, 7256.0625000
2: -7269.7172852, 5313.2114258, -3383.4587402, 2517.0407715, -9786.7558594, 8696.6699219
3: -2802.5759277, 7189.0590820, -1326.4201660, 3353.4309082, -6156.0063477, 8515.4794922
4: -8069.6728516, 5236.5312500, -3765.1281738, 2475.0305176, -10544.7011719, 9001.6591797

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4292608, upper bound: 4551.6875449
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4343192, upper bound: 4551.7947080
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6346.4194336, 5103.5146484, -2952.6821289, 2402.8015137, -8749.2207031, 8056.1967773
1: -5092.1015625, 4907.2846680, -2365.3845215, 2314.5366211, -7406.6381836, 7272.6689453
2: -7292.6235352, 5332.8779297, -3383.4587402, 2517.0407715, -9809.6630859, 8716.3349609
3: -2813.4733887, 7209.9868164, -1326.4201660, 3353.4309082, -6166.9042969, 8536.4072266
4: -8097.8183594, 5257.0014648, -3765.1281738, 2475.0305176, -10572.8486328, 9022.1298828

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7427411, upper bound: 4551.6863623
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7477994, upper bound: 4551.7935255
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6366.1660156, 5119.2890625, -4267.7421875, 3427.3356934, -9793.5019531, 9387.0302734
1: -5110.7729492, 4923.2451172, -3433.8293457, 3298.9086914, -8409.6816406, 8357.0742188
2: -7322.4482422, 5348.0849609, -4925.3828125, 3575.4016113, -10897.8486328, 10273.4677734
3: -2820.9536133, 7239.8808594, -1881.0106201, 4863.0942383, -7676.4799805, 9120.8203125
4: -8126.7597656, 5270.9204102, -5456.5566406, 3519.1264648, -11645.8867188, 10727.4765625

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4241378, upper bound: 4551.7171103
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4219976, upper bound: 4551.7437677
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6390.3413086, 5137.2963867, -4267.7421875, 3427.3356934, -9817.6767578, 9405.0390625
1: -5128.1879883, 4939.5961914, -3433.8293457, 3298.9086914, -8427.0957031, 8373.4257812
2: -7344.9951172, 5367.6391602, -4925.3828125, 3575.4016113, -10920.3964844, 10293.0214844
3: -2831.8232422, 7260.3583984, -1881.0106201, 4863.0942383, -7687.5229492, 9141.3691406
4: -8154.4711914, 5291.2568359, -5456.5566406, 3519.1264648, -11673.5957031, 10747.8134766

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7376181, upper bound: 4551.7160190
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7354778, upper bound: 4551.7425851
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -5158.3984375, 4167.5395508, -8764.6572266, 8865.9423828
1: -3687.0437012, 3569.1484375, -4135.0488281, 4011.9240723, -7698.9667969, 7704.1972656
2: -5276.2031250, 3879.7956543, -5915.9174805, 4360.9184570, -9632.9892578, 9791.7519531
3: -2047.8087158, 5221.5805664, -2300.4028320, 5857.5903320, -7904.4311523, 7519.7099609
4: -5870.9716797, 3822.0715332, -6584.4262695, 4295.7910156, -10160.3740234, 10400.6279297

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1621650, upper bound: 4552.1621650
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1621650, upper bound: 4552.1916969
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -5158.3984375, 4167.5395508, -9271.5009766, 9281.6806641
1: -4091.2307129, 3969.2382812, -4135.0488281, 4011.9240723, -8103.1542969, 8104.2871094
2: -5853.0830078, 4314.3935547, -5915.9174805, 4360.9184570, -10211.1474609, 10227.9863281
3: -2275.4304199, 5795.1562500, -2300.4028320, 5857.5903320, -8133.0200195, 8094.7338867
4: -6514.6850586, 4250.0151367, -6584.4262695, 4295.7910156, -10805.3154297, 10829.7656250

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1916969, upper bound: 4552.1621650
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1916969, upper bound: 4552.1916969
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -6412.9111328, 5157.1645508, -9749.0966797, 10120.4550781
1: -3687.0437012, 3569.1484375, -5145.5615234, 4959.1176758, -8643.9384766, 8714.7089844
2: -5276.2031250, 3879.7956543, -7368.8452148, 5389.6997070, -10655.1494141, 11239.4296875
3: -2047.8087158, 5221.5805664, -2843.5869141, 7284.7187500, -9311.7509766, 8059.4125977
4: -5870.9716797, 3822.0715332, -8182.3857422, 5312.6733398, -11169.6845703, 11998.6250000

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1545275, upper bound: 4551.4742097
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1513498, upper bound: 4552.1940604
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -6412.6123047, 5156.9331055, -10257.2060547, 10535.8955078
1: -4091.2307129, 3969.2382812, -5145.3183594, 4958.8945312, -9048.7138672, 9114.5566406
2: -5853.0830078, 4314.3935547, -7368.4902344, 5389.4604492, -11233.0703125, 11675.3134766
3: -2275.4304199, 5795.1562500, -2843.4611816, 7284.3759766, -9540.1865234, 8634.3105469
4: -6514.6850586, 4250.0151367, -8182.0014648, 5312.4379883, -11814.3906250, 12427.3789062

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1794140, upper bound: 4551.4667226
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1762363, upper bound: 4552.1865733
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6320.0683594, 5083.7685547, -5158.3984375, 4167.5395508, -10487.4238281, 10235.7500000
1: -5072.7436523, 4889.1918945, -4135.0488281, 4011.9240723, -9084.6679688, 9020.1181641
2: -7267.3095703, 5311.6176758, -5915.9174805, 4360.9184570, -11617.0273438, 11214.5488281
3: -2801.7348633, 7186.7387695, -2300.4028320, 5857.5903320, -8652.4746094, 9466.6484375
4: -8067.0654297, 5234.9580078, -6584.4262695, 4295.7910156, -12355.1386719, 11803.2910156

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4742097, upper bound: 4552.1554065
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4667226, upper bound: 4552.1802930
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6344.4028320, 5101.9619141, -5158.3984375, 4167.5395508, -10511.9404297, 10253.6708984
1: -5090.4521484, 4905.7983398, -4135.0488281, 4011.9240723, -9102.3759766, 9036.7890625
2: -7290.2241211, 5331.2797852, -5915.9174805, 4360.9184570, -11643.9433594, 11234.2978516
3: -2812.6291504, 7207.6782227, -2300.4028320, 5857.5903320, -8663.5419922, 9488.0468750
4: -8095.2226562, 5255.4267578, -6584.4262695, 4295.7910156, -12387.0410156, 11823.4814453

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1940604, upper bound: 4552.1529031
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1865733, upper bound: 4552.1777897
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6366.1660156, 5119.2890625, -6465.6679688, 5197.9145508, -11557.8154297, 11577.7519531
1: -5110.7729492, 4923.2451172, -5188.6606445, 4998.1103516, -10105.2763672, 10107.7431641
2: -7322.4482422, 5348.0849609, -7431.6689453, 5431.6601562, -12735.1406250, 12760.1982422
3: -2820.9536133, 7239.8808594, -2865.7429199, 7345.2402344, -10138.3300781, 10080.5986328
4: -8126.7597656, 5270.9204102, -8250.3779297, 5354.0131836, -13465.0605469, 13504.7070312

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4646321, upper bound: 4551.4668377
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1872591
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6390.3413086, 5137.2963867, -6465.6679688, 5197.9145508, -11584.8183594, 11595.4892578
1: -5128.1879883, 4939.5961914, -5188.6606445, 4998.1103516, -10125.4355469, 10124.1347656
2: -7344.9951172, 5367.6391602, -7431.6689453, 5431.6601562, -12761.6904297, 12779.8447266
3: -2831.8232422, 7260.3583984, -2865.7429199, 7345.2402344, -10149.3730469, 10101.5341797
4: -8154.4711914, 5291.2568359, -8250.3779297, 5354.0131836, -13496.5107422, 13524.7314453

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844829, upper bound: 4551.4668377
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844829, upper bound: 4552.1872591
time: 0.88 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.42 seconds
IS_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.42
Output dim: 0, lower bound: -4551.7380623, upper bound: 4551.7103340
IS_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.42
Output dim: 0, lower bound: -4551.7401350, upper bound: 4551.8036859
IS_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.42
Output dim: 0, lower bound: -4551.7679201, upper bound: 4551.7029061
IS_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.42
Output dim: 0, lower bound: -4551.7699927, upper bound: 4551.7962579
IS_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.42
Output dim: 0, lower bound: -4551.7322538, upper bound: 4551.7398922
IS_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.42
Output dim: 0, lower bound: -4551.7280324, upper bound: 4551.7652860
IS_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.42
Output dim: 0, lower bound: -4551.7621116, upper bound: 4551.7324642
IS_A2_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.42
Output dim: 0, lower bound: -4551.7578902, upper bound: 4551.7578580
IS_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.42
Output dim: 0, lower bound: -4550.4292608, upper bound: 4551.6875449
IS_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.42
Output dim: 0, lower bound: -4550.4343192, upper bound: 4551.7947080
IS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.42
Output dim: 0, lower bound: -4551.7427411, upper bound: 4551.6863623
IS_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.42
Output dim: 0, lower bound: -4551.7477994, upper bound: 4551.7935255
IS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.42
Output dim: 0, lower bound: -4550.4241378, upper bound: 4551.7171103
IS_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.42
Output dim: 0, lower bound: -4550.4219976, upper bound: 4551.7437677
IS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.42
Output dim: 0, lower bound: -4551.7376181, upper bound: 4551.7160190
IS_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.42
Output dim: 0, lower bound: -4551.7354778, upper bound: 4551.7425851
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.42
Output dim: 0, lower bound: -4552.1621650, upper bound: 4552.1621650
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.42
Output dim: 0, lower bound: -4552.1621650, upper bound: 4552.1916969
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.42
Output dim: 0, lower bound: -4552.1916969, upper bound: 4552.1621650
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.42
Output dim: 0, lower bound: -4552.1916969, upper bound: 4552.1916969
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.42
Output dim: 0, lower bound: -4552.1545275, upper bound: 4551.4742097
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.42
Output dim: 0, lower bound: -4552.1513498, upper bound: 4552.1940604
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.42
Output dim: 0, lower bound: -4552.1794140, upper bound: 4551.4667226
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.42
Output dim: 0, lower bound: -4552.1762363, upper bound: 4552.1865733
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.42
Output dim: 0, lower bound: -4551.4742097, upper bound: 4552.1554065
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.42
Output dim: 0, lower bound: -4551.4667226, upper bound: 4552.1802930
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.42
Output dim: 0, lower bound: -4552.1940604, upper bound: 4552.1529031
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.42
Output dim: 0, lower bound: -4552.1865733, upper bound: 4552.1777897
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.42
Output dim: 0, lower bound: -4551.4646321, upper bound: 4551.4668377
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.42
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1872591
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.42
Output dim: 0, lower bound: -4552.1844829, upper bound: 4551.4668377
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.42
Output dim: 0, lower bound: -4552.1844829, upper bound: 4552.1872591

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -4597.1176758, 3707.5437012, -8304.6611328, 8304.6611328
1: -3687.0437012, 3569.1484375, -3687.0437012, 3569.1484375, -7256.1918945, 7256.1918945
2: -5276.2031250, 3879.7956543, -5276.2031250, 3879.7956543, -9150.5644531, 9150.5644531
3: -2047.8087158, 5221.5805664, -2047.8087158, 5221.5805664, -7266.1542969, 7266.1542969
4: -5870.9716797, 3822.0715332, -5870.9716797, 3822.0715332, -9685.8447266, 9685.8437500

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.7139066, upper bound: 4551.7342625
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7323025, upper bound: 4551.7397307
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -5103.9614258, 4123.2832031, -8720.4003906, 8811.5039062
1: -3687.0437012, 3569.1484375, -4091.2307129, 3969.2382812, -7656.2822266, 7660.3789062
2: -5276.2031250, 3879.7956543, -5853.0830078, 4314.3935547, -9586.7988281, 9728.7226562
3: -2047.8087158, 5221.5805664, -2275.4304199, 5795.1562500, -7841.1791992, 7494.9267578
4: -5870.9716797, 3822.0715332, -6514.6850586, 4250.0151367, -10114.9833984, 10330.7871094

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.7139066, upper bound: 4551.7641203
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7323025, upper bound: 4551.7695884
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -4597.1176758, 3707.5437012, -8811.5029297, 8720.4003906
1: -4091.2307129, 3969.2382812, -3687.0437012, 3569.1484375, -7660.3789062, 7656.2822266
2: -5853.0830078, 4314.3935547, -5276.2031250, 3879.7956543, -9728.7226562, 9586.7988281
3: -2275.4304199, 5795.1562500, -2047.8087158, 5221.5805664, -7494.9267578, 7841.1787109
4: -6514.6850586, 4250.0151367, -5870.9716797, 3822.0715332, -10330.7861328, 10114.9833984

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6117698, upper bound: 4551.7330142
time: 1.06 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7621604, upper bound: 4551.7323027
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -5103.9614258, 4123.2832031, -9227.2421875, 9227.2431641
1: -4091.2307129, 3969.2382812, -4091.2307129, 3969.2382812, -8060.4687500, 8060.4687500
2: -5853.0830078, 4314.3935547, -5853.0830078, 4314.3935547, -10164.9560547, 10164.9570312
3: -2275.4304199, 5795.1562500, -2275.4304199, 5795.1562500, -8069.9511719, 8069.9506836
4: -6514.6850586, 4250.0151367, -6514.6850586, 4250.0151367, -10759.9257812, 10759.9248047

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6117698, upper bound: 4551.7623046
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7621604, upper bound: 4551.7590118
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -6312.8598633, 5078.2158203, -9667.3505859, 10019.4794922
1: -3687.0437012, 3569.1484375, -5066.8012695, 4883.8798828, -8565.9033203, 8635.9492188
2: -5276.2031250, 3879.7956543, -7258.7055664, 5305.9140625, -10567.6816406, 11126.1591797
3: -2047.8087158, 5221.5805664, -2798.7263184, 7178.4555664, -9204.9589844, 8011.2045898
4: -5870.9716797, 3822.0715332, -8057.7509766, 5229.3339844, -11082.9003906, 11871.3505859

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4445115, upper bound: 4550.4219729
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4968185, upper bound: 4550.4290304
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -6337.2187500, 5096.4145508, -9685.2753906, 10044.7617188
1: -3687.0437012, 3569.1484375, -5084.6010742, 4900.4902344, -8582.5839844, 8653.7500000
2: -5276.2031250, 3879.7956543, -7281.6796875, 5325.5659180, -10587.4199219, 11153.1357422
3: -2047.8087158, 5221.5805664, -2809.6096191, 7199.4492188, -9226.4169922, 8022.2612305
4: -5870.9716797, 3822.0715332, -8085.9736328, 5249.7993164, -11103.0986328, 11903.3212891

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4444965, upper bound: 4551.7388089
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7091438, upper bound: 4551.7470782
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -6312.5610352, 5077.9853516, -10175.4648438, 10435.8437500
1: -4091.2307129, 3969.2382812, -5066.5561523, 4883.6586914, -8970.6796875, 9035.7949219
2: -5853.0830078, 4314.3935547, -7258.3481445, 5305.6767578, -11145.6035156, 11562.0419922
3: -2275.4304199, 5795.1562500, -2798.6015625, 7178.1118164, -9433.3994141, 8586.1044922
4: -6514.6850586, 4250.0151367, -8057.3652344, 5229.1005859, -11727.6123047, 12300.1035156

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.1823506, upper bound: 4550.4219544
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4984060, upper bound: 4550.4219561
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -6336.9213867, 5096.1840820, -10193.3916016, 10460.2021484
1: -4091.2307129, 3969.2382812, -5084.3613281, 4900.2700195, -8987.3593750, 9053.5996094
2: -5853.0830078, 4314.3935547, -7281.3261719, 5325.3300781, -11165.3427734, 11589.0234375
3: -2275.4304199, 5795.1562500, -2809.4848633, 7199.1093750, -9454.8544922, 8597.1582031
4: -6514.6850586, 4250.0151367, -8085.5932617, 5249.5659180, -11747.8085938, 12332.0800781

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.5933311, upper bound: 4551.7388911
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7376065, upper bound: 4551.7391450
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6312.8598633, 5078.2158203, -4597.1176758, 3707.5437012, -10019.4804688, 9667.3505859
1: -5066.8012695, 4883.8798828, -3687.0437012, 3569.1484375, -8635.9492188, 8565.9033203
2: -7258.7055664, 5305.9140625, -5276.2031250, 3879.7956543, -11126.1591797, 10567.6816406
3: -2798.7263184, 7178.4555664, -2047.8087158, 5221.5805664, -8011.2045898, 9204.9589844
4: -8057.7509766, 5229.3339844, -5870.9716797, 3822.0715332, -11871.3505859, 11082.9003906

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.8832384, upper bound: 4551.7139337
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4302495, upper bound: 4551.7125020
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6312.5610352, 5077.9853516, -5103.9614258, 4123.2832031, -10435.8427734, 10175.4638672
1: -5066.5561523, 4883.6586914, -4091.2307129, 3969.2382812, -9035.7949219, 8970.6796875
2: -7258.3481445, 5305.6767578, -5853.0830078, 4314.3935547, -11562.0419922, 11145.6035156
3: -2798.6015625, 7178.1118164, -2275.4304199, 5795.1562500, -8586.1044922, 9433.3994141
4: -8057.3652344, 5229.1005859, -6514.6850586, 4250.0151367, -12300.1044922, 11727.6123047

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.8753052, upper bound: 4551.7423962
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4223163, upper bound: 4551.7409645
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6337.2187500, 5096.4145508, -4597.1176758, 3707.5437012, -10044.7617188, 9685.2753906
1: -5084.6010742, 4900.4902344, -3687.0437012, 3569.1484375, -8653.7500000, 8582.5849609
2: -7281.6796875, 5325.5659180, -5276.2031250, 3879.7956543, -11153.1357422, 10587.4199219
3: -2809.6096191, 7199.4492188, -2047.8087158, 5221.5805664, -8022.2617188, 9226.4169922
4: -8085.9736328, 5249.7993164, -5870.9716797, 3822.0715332, -11903.3212891, 11103.0976562

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6049666, upper bound: 4551.7124846
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7470782, upper bound: 4551.7124870
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6336.9213867, 5096.1840820, -5103.9614258, 4123.2832031, -10460.2021484, 10193.3916016
1: -5084.3613281, 4900.2700195, -4091.2307129, 3969.2382812, -9053.5996094, 8987.3593750
2: -7281.3261719, 5325.3300781, -5853.0830078, 4314.3935547, -11589.0244141, 11165.3427734
3: -2809.4848633, 7199.1093750, -2275.4304199, 5795.1562500, -8597.1582031, 9454.8554688
4: -8085.5932617, 5249.5659180, -6514.6850586, 4250.0151367, -12332.0800781, 11747.8095703

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.5970334, upper bound: 4551.7409471
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7391450, upper bound: 4551.7409495
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6366.1660156, 5119.2890625, -6390.3413086, 5137.2963867, -11494.0605469, 11503.3271484
1: -5110.7729492, 4923.2451172, -5128.1879883, 4939.5961914, -10043.9892578, 10047.7578125
2: -7322.4482422, 5348.0849609, -7344.9951172, 5367.6391602, -12667.4521484, 12674.3535156
3: -2820.9536133, 7239.8808594, -2831.8232422, 7260.3583984, -10053.3544922, 10043.4609375
4: -8126.7597656, 5270.9204102, -8154.4711914, 5291.2568359, -13398.4580078, 13409.8847656

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.7561070, upper bound: 4551.7420811
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4258744, upper bound: 4551.7417620
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6390.3413086, 5137.2963867, -6366.1660156, 5119.2890625, -11503.3271484, 11494.0615234
1: -5128.1879883, 4939.5961914, -5110.7729492, 4923.2451172, -10047.7568359, 10043.9892578
2: -7344.9951172, 5367.6391602, -7322.4482422, 5348.0849609, -12674.3544922, 12667.4521484
3: -2831.8232422, 7260.3583984, -2820.9536133, 7239.8808594, -10043.4609375, 10053.3535156
4: -8154.4711914, 5291.2568359, -8126.7597656, 5270.9204102, -13409.8847656, 13398.4580078

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.5549226, upper bound: 4550.4258699
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.5417017, upper bound: 4550.4273814
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6390.3413086, 5137.2963867, -6390.3413086, 5137.2963867, -11521.0634766, 11521.0644531
1: -5128.1879883, 4939.5961914, -5128.1879883, 4939.5961914, -10064.1494141, 10064.1494141
2: -7344.9951172, 5367.6391602, -7344.9951172, 5367.6391602, -12694.0019531, 12694.0019531
3: -2831.8232422, 7260.3583984, -2831.8232422, 7260.3583984, -10064.3964844, 10064.3964844
4: -8154.4711914, 5291.2568359, -8154.4711914, 5291.2568359, -13429.9082031, 13429.9091797

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.5549222, upper bound: 4551.7333355
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.5417018, upper bound: 4551.7375208
time: 0.86 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.91 seconds
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.91
Output dim: 0, lower bound: -4550.7139066, upper bound: 4551.7342625
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.91
Output dim: 0, lower bound: -4551.7323025, upper bound: 4551.7397307
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.91
Output dim: 0, lower bound: -4550.7139066, upper bound: 4551.7641203
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.91
Output dim: 0, lower bound: -4551.7323025, upper bound: 4551.7695884
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.91
Output dim: 0, lower bound: -4551.6117698, upper bound: 4551.7330142
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.91
Output dim: 0, lower bound: -4551.7621604, upper bound: 4551.7323027
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.91
Output dim: 0, lower bound: -4551.6117698, upper bound: 4551.7623046
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.91
Output dim: 0, lower bound: -4551.7621604, upper bound: 4551.7590118
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.91
Output dim: 0, lower bound: -4550.4445115, upper bound: 4550.4219729
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.91
Output dim: 0, lower bound: -4551.4968185, upper bound: 4550.4290304
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.91
Output dim: 0, lower bound: -4550.4444965, upper bound: 4551.7388089
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.91
Output dim: 0, lower bound: -4551.7091438, upper bound: 4551.7470782
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.91
Output dim: 0, lower bound: -4551.1823506, upper bound: 4550.4219544
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.91
Output dim: 0, lower bound: -4551.4984060, upper bound: 4550.4219561
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.91
Output dim: 0, lower bound: -4551.5933311, upper bound: 4551.7388911
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.91
Output dim: 0, lower bound: -4551.7376065, upper bound: 4551.7391450
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.91
Output dim: 0, lower bound: -4550.8832384, upper bound: 4551.7139337
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.91
Output dim: 0, lower bound: -4550.4302495, upper bound: 4551.7125020
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.91
Output dim: 0, lower bound: -4550.8753052, upper bound: 4551.7423962
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.91
Output dim: 0, lower bound: -4550.4223163, upper bound: 4551.7409645
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.91
Output dim: 0, lower bound: -4551.6049666, upper bound: 4551.7124846
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.91
Output dim: 0, lower bound: -4551.7470782, upper bound: 4551.7124870
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.91
Output dim: 0, lower bound: -4551.5970334, upper bound: 4551.7409471
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.91
Output dim: 0, lower bound: -4551.7391450, upper bound: 4551.7409495
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.91
Output dim: 0, lower bound: -4550.7561070, upper bound: 4551.7420811
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.91
Output dim: 0, lower bound: -4550.4258744, upper bound: 4551.7417620
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.91
Output dim: 0, lower bound: -4551.5549226, upper bound: 4550.4258699
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.91
Output dim: 0, lower bound: -4551.5417017, upper bound: 4550.4273814
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.91
Output dim: 0, lower bound: -4551.5549222, upper bound: 4551.7333355
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.91
Output dim: 0, lower bound: -4551.5417018, upper bound: 4551.7375208
Binary search (step 1): status=Status.VERIFIED, low=0.7500000, high=1.0000000, mid=0.7500000, abs_max=5687.5751953125
rel_dist={0: [-4552.289001211694, 4552.289001211693]}

## Binary search (step 2) starts
Candidate diff: 0.8750000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2865166, upper bound: 4552.2783404
time: 0.71 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711
time: 0.98 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.84 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.84
Output dim: 0, lower bound: -4552.2865166, upper bound: 4552.2783404
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.84
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -2987.9899902, 2430.9243164, -3136.6748047, 2550.9006348, -5538.8906250, 5567.5991211
1: -2393.8061523, 2341.6025391, -2513.2941895, 2457.4460449, -4851.2519531, 4854.8964844
2: -3424.2626953, 2546.4648438, -3595.4755859, 2672.1398926, -6096.4023438, 6141.9404297
3: -1341.7878418, 3393.8977051, -1407.5072021, 3564.2558594, -4906.0439453, 4801.4047852
4: -3810.2302246, 2503.8750000, -3999.4067383, 2627.0041504, -6437.2343750, 6503.2817383

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711
time: 0.87 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711
time: 0.75 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -5188.6577148, 4191.8315430, -3136.6748047, 2550.9006348, -7739.5585938, 7328.5063477
1: -4159.4165039, 4035.3022461, -2513.2941895, 2457.4460449, -6616.8623047, 6548.5966797
2: -5950.7929688, 4386.3300781, -3595.4755859, 2672.1398926, -8622.9326172, 7981.8056641
3: -2313.8149414, 5892.1474609, -1407.5072021, 3564.2558594, -5878.0708008, 7299.6542969
4: -6623.0805664, 4320.7236328, -3999.4067383, 2627.0041504, -9250.0830078, 8320.1308594

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711
time: 0.90 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711
time: 0.76 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.79 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.79
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.79
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.79
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.79
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -2987.9899902, 2430.9243164, -2987.9899902, 2430.9243164, -5418.9140625, 5418.9140625
1: -2393.8061523, 2341.6025391, -2393.8061523, 2341.6025391, -4735.4086914, 4735.4086914
2: -3424.2626953, 2546.4648438, -3424.2626953, 2546.4648438, -5970.7275391, 5970.7275391
3: -1341.7878418, 3393.8977051, -1341.7878418, 3393.8977051, -4735.6855469, 4735.6855469
4: -3810.2302246, 2503.8750000, -3810.2302246, 2503.8750000, -6314.1054688, 6314.1054688

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2085840, upper bound: 4552.1951258
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1880421
time: 0.81 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -2987.9899902, 2430.9243164, -5188.6577148, 4191.8315430, -7179.8212891, 7619.5820312
1: -2393.8061523, 2341.6025391, -4159.4165039, 4035.3022461, -6429.1083984, 6501.0190430
2: -3424.2626953, 2546.4648438, -5950.7929688, 4386.3300781, -7810.5922852, 8497.2558594
3: -1341.7878418, 3393.8977051, -2313.8149414, 5892.1474609, -7233.9355469, 5707.7128906
4: -3810.2302246, 2503.8750000, -6623.0805664, 4320.7236328, -8130.9526367, 9126.9550781

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2085840, upper bound: 4552.1951258
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1880421
time: 0.91 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -5188.6577148, 4191.8315430, -2987.9899902, 2430.9243164, -7619.5820312, 7179.8212891
1: -4159.4165039, 4035.3022461, -2393.8061523, 2341.6025391, -6501.0190430, 6429.1083984
2: -5950.7929688, 4386.3300781, -3424.2626953, 2546.4648438, -8497.2558594, 7810.5922852
3: -2313.8149414, 5892.1474609, -1341.7878418, 3393.8977051, -5707.7128906, 7233.9355469
4: -6623.0805664, 4320.7236328, -3810.2302246, 2503.8750000, -9126.9550781, 8130.9531250

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1946649
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1905380
time: 0.78 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -5188.6577148, 4191.8315430, -5188.6577148, 4191.8315430, -9380.4892578, 9380.4892578
1: -4159.4165039, 4035.3022461, -4159.4165039, 4035.3022461, -8194.7187500, 8194.7187500
2: -5950.7929688, 4386.3300781, -5950.7929688, 4386.3300781, -10334.5771484, 10334.5771484
3: -2313.8149414, 5892.1474609, -2313.8149414, 5892.1474609, -8205.9628906, 8205.9628906
4: -6623.0805664, 4320.7236328, -6623.0805664, 4320.7236328, -10938.7910156, 10938.7919922

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1946649
time: 1.04 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1905380
time: 0.76 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.47 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 0, lower bound: -4552.2085840, upper bound: 4552.1951258
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1880421
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 0, lower bound: -4552.2085840, upper bound: 4552.1951258
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1880421
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1946649
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1905380
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1946649
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1905380

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -2987.9899902, 2430.9243164, -5383.6064453, 5390.7915039
1: -2365.3845215, 2314.5366211, -2393.8061523, 2341.6025391, -4706.9873047, 4708.3427734
2: -3383.4587402, 2517.0407715, -3424.2626953, 2546.4648438, -5929.9228516, 5941.3027344
3: -1326.4201660, 3353.4309082, -1341.7878418, 3393.8977051, -4720.3173828, 4695.2187500
4: -3765.1281738, 2475.0305176, -3810.2302246, 2503.8750000, -6269.0029297, 6285.2602539

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1859666, upper bound: 4552.1859666
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1859666, upper bound: 4552.1880421
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4214.6674805, 3387.0080566, -2987.4873047, 2430.5163574, -6645.1831055, 6374.4951172
1: -3389.9724121, 3259.9172363, -2393.4016113, 2341.2058105, -5731.1782227, 5653.3188477
2: -4861.8188477, 3533.9157715, -3423.6840820, 2546.0341797, -7407.8530273, 6957.5991211
3: -1859.8743896, 4801.1406250, -1341.5629883, 3393.3225098, -5253.1967773, 6142.7036133
4: -5387.0976562, 3478.8435059, -3809.5886230, 2503.4548340, -7890.5522461, 7288.4321289

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1859666
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1880421
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -5188.6577148, 4191.8315430, -7144.5136719, 7591.4589844
1: -2365.3845215, 2314.5366211, -4159.4165039, 4035.3022461, -6400.6865234, 6473.9531250
2: -3383.4587402, 2517.0407715, -5950.7929688, 4386.3300781, -7769.7875977, 8467.8320312
3: -1326.4201660, 3353.4309082, -2313.8149414, 5892.1474609, -7218.5668945, 5667.2460938
4: -3765.1281738, 2475.0305176, -6623.0805664, 4320.7236328, -8085.8510742, 9098.1093750

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1884626, upper bound: 4552.1856700
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1884626, upper bound: 4552.1880421
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4212.4501953, 3385.3171387, -5188.1044922, 4191.3837891, -8403.8330078, 8573.4218750
1: -3388.1425781, 3258.2839355, -4158.9716797, 4034.8710938, -7423.0136719, 7417.2558594
2: -4859.1650391, 3532.1770020, -5950.1562500, 4385.8618164, -9245.0263672, 9482.3330078
3: -1858.9870605, 4798.5488281, -2313.5698242, 5891.5151367, -7750.5019531, 7112.1186523
4: -5384.1958008, 3477.1557617, -6622.3745117, 4320.2626953, -9704.4589844, 10099.5302734

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905380, upper bound: 4552.1856700
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905380, upper bound: 4552.1880421
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -2987.9899902, 2430.9243164, -7589.3227539, 7155.5288086
1: -4135.0488281, 4011.9240723, -2393.8061523, 2341.6025391, -6476.6513672, 6405.7290039
2: -5915.9174805, 4360.9184570, -3424.2626953, 2546.4648438, -8462.3818359, 7785.1806641
3: -2300.4028320, 5857.5903320, -1341.7878418, 3393.8977051, -5694.3007812, 7199.3779297
4: -6584.4262695, 4295.7910156, -3810.2302246, 2503.8750000, -9088.3007812, 8106.0195312

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1884626
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1905380
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6436.9165039, 5175.7587891, -2987.4873047, 2430.5163574, -8867.4326172, 8163.2456055
1: -5165.1123047, 4976.9003906, -2393.4016113, 2341.2058105, -7506.3183594, 7370.3017578
2: -7397.4135742, 5408.8471680, -3423.6840820, 2546.0341797, -9943.4462891, 8832.5312500
3: -2853.6982422, 7312.2685547, -1341.5629883, 3393.3225098, -6247.0205078, 8653.8300781
4: -8213.3105469, 5331.5375977, -3809.5886230, 2503.4548340, -10716.7656250, 9141.1250000

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1884626
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1905380
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -5188.6577148, 4191.8315430, -9350.2294922, 9356.1972656
1: -4135.0488281, 4011.9240723, -4159.4165039, 4035.3022461, -8170.3510742, 8171.3408203
2: -5915.9174805, 4360.9184570, -5950.7929688, 4386.3300781, -10299.6093750, 10309.1445312
3: -2300.4028320, 5857.5903320, -2313.8149414, 5892.1474609, -8192.5507812, 8171.4052734
4: -6584.4262695, 4295.7910156, -6623.0805664, 4320.7236328, -10900.1123047, 10913.8359375

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1881660
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1905380
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6435.2099609, 5174.4443359, -5188.1044922, 4191.3837891, -10626.5937500, 10358.9414062
1: -5163.7192383, 4975.6435547, -4158.9716797, 4034.8710938, -9198.5869141, 9133.2871094
2: -7395.3833008, 5407.4941406, -5950.1562500, 4385.8618164, -11772.8632812, 11348.4228516
3: -2852.9838867, 7310.3134766, -2313.5698242, 5891.5151367, -8741.1455078, 9603.6289062
4: -8211.1152344, 5330.2036133, -6622.3745117, 4320.2626953, -12526.2080078, 11939.9511719

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1881660
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1905380
time: 0.82 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.47 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.47
Output dim: 0, lower bound: -4552.1859666, upper bound: 4552.1859666
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.47
Output dim: 0, lower bound: -4552.1859666, upper bound: 4552.1880421
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.47
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1859666
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.47
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1880421
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.47
Output dim: 0, lower bound: -4552.1884626, upper bound: 4552.1856700
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.47
Output dim: 0, lower bound: -4552.1884626, upper bound: 4552.1880421
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.47
Output dim: 0, lower bound: -4552.1905380, upper bound: 4552.1856700
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.47
Output dim: 0, lower bound: -4552.1905380, upper bound: 4552.1880421
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.47
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1884626
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.47
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1905380
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.47
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1884626
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.47
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1905380
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.47
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1881660
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.47
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1905380
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.47
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1881660
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.47
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1905380

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -2952.6821289, 2402.8015137, -5355.4833984, 5355.4833984
1: -2365.3845215, 2314.5366211, -2365.3845215, 2314.5366211, -4679.9208984, 4679.9208984
2: -3383.4587402, 2517.0407715, -3383.4587402, 2517.0407715, -5900.4965820, 5900.4970703
3: -1326.4201660, 3353.4309082, -1326.4201660, 3353.4309082, -4679.8505859, 4679.8505859
4: -3765.1281738, 2475.0305176, -3765.1281738, 2475.0305176, -6240.1582031, 6240.1582031

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6986709, upper bound: 4551.7638377
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7920228, upper bound: 4551.7659535
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -4199.0942383, 3375.1484375, -6327.8305664, 6601.8955078
1: -2365.3845215, 2314.5366211, -3377.1232910, 3248.4418945, -5613.8261719, 5691.6601562
2: -3383.4587402, 2517.0407715, -4843.1782227, 3521.7045898, -6905.1625977, 7360.2182617
3: -1326.4201660, 3353.4309082, -1853.6423340, 4782.9531250, -6109.3725586, 5207.0732422
4: -3765.1281738, 2475.0305176, -5366.7207031, 3466.9887695, -7232.1166992, 7841.7509766

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6986709, upper bound: 4551.7638377
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7920228, upper bound: 4551.7659535
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4199.0942383, 3375.1484375, -2952.6821289, 2402.8015137, -6601.8955078, 6327.8305664
1: -3377.1232910, 3248.4418945, -2365.3845215, 2314.5366211, -5691.6601562, 5613.8261719
2: -4843.1782227, 3521.7045898, -3383.4587402, 2517.0407715, -7360.2172852, 6905.1625977
3: -1853.6423340, 4782.9531250, -1326.4201660, 3353.4309082, -5207.0732422, 6109.3725586
4: -5366.7207031, 3466.9887695, -3765.1281738, 2475.0305176, -7841.7509766, 7232.1166992

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7282293, upper bound: 4551.7583069
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7536229, upper bound: 4551.7536229
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4267.7421875, 3427.3356934, -4267.7421875, 3427.3356934, -7695.0771484, 7695.0771484
1: -3433.8293457, 3298.9086914, -3433.8293457, 3298.9086914, -6732.7377930, 6732.7377930
2: -4925.3828125, 3575.4016113, -4925.3828125, 3575.4016113, -8500.7832031, 8500.7841797
3: -1881.0106201, 4863.0942383, -1881.0106201, 4863.0942383, -6744.1049805, 6744.1049805
4: -5456.5566406, 3519.1264648, -5456.5566406, 3519.1264648, -8975.6835938, 8975.6835938

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7282293, upper bound: 4551.7583069
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7536229, upper bound: 4551.7536229
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -5158.3984375, 4167.5395508, -7120.2216797, 7561.2001953
1: -2365.3845215, 2314.5366211, -4135.0488281, 4011.9240723, -6377.3085938, 6449.5854492
2: -3383.4587402, 2517.0407715, -5915.9174805, 4360.9184570, -7744.3754883, 8432.9580078
3: -1326.4201660, 3353.4309082, -2300.4028320, 5857.5903320, -7184.0102539, 5653.8339844
4: -3765.1281738, 2475.0305176, -6584.4262695, 4295.7910156, -8060.9179688, 9059.4570312

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7052373, upper bound: 4551.7637713
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7985891, upper bound: 4551.7658871
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -6424.9907227, 5166.5434570, -8119.2255859, 8827.7910156
1: -2365.3845215, 2314.5366211, -5155.3681641, 4968.0869141, -7333.4716797, 7469.9047852
2: -3383.4587402, 2517.0407715, -7383.2236328, 5399.3579102, -8782.8154297, 9900.2636719
3: -1326.4201660, 3353.4309082, -2848.6857910, 7298.5976562, -8625.0175781, 6202.1166992
4: -3765.1281738, 2475.0305176, -8197.9521484, 5322.1879883, -9087.3164062, 10672.9824219

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7052373, upper bound: 4551.7638109
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7985891, upper bound: 4551.7659267
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4196.5258789, 3373.1972656, -5158.3984375, 4167.5395508, -8364.0644531, 8531.5957031
1: -3375.0051270, 3246.5498047, -4135.0488281, 4011.9240723, -7386.9291992, 7381.5981445
2: -4840.1069336, 3519.6916504, -5915.9174805, 4360.9184570, -9201.0234375, 9435.6093750
3: -1852.6146240, 4779.9594727, -2300.4028320, 5857.5903320, -7710.2045898, 7080.3623047
4: -5363.3623047, 3465.0339355, -6584.4262695, 4295.7910156, -9659.1513672, 10049.4599609

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7347956, upper bound: 4551.7582405
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7601892, upper bound: 4551.7535565
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4267.7421875, 3427.3356934, -6465.6679688, 5197.9145508, -9465.6562500, 9893.0039062
1: -3433.8293457, 3298.9086914, -5188.6606445, 4998.1103516, -8431.9394531, 8487.5683594
2: -4925.3828125, 3575.4016113, -7431.6689453, 5431.6601562, -10357.0410156, 11007.0703125
3: -1881.0106201, 4863.0942383, -2865.7429199, 7345.2402344, -9226.2509766, 7724.6606445
4: -5456.5566406, 3519.1264648, -8250.3779297, 5354.0131836, -10810.5673828, 11769.5009766

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7347956, upper bound: 4551.7582713
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7601892, upper bound: 4551.7535565
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -2952.6821289, 2402.8015137, -7561.2001953, 7120.2216797
1: -4135.0488281, 4011.9240723, -2365.3845215, 2314.5366211, -6449.5854492, 6377.3085938
2: -5915.9174805, 4360.9184570, -3383.4587402, 2517.0407715, -8432.9580078, 7744.3750000
3: -2300.4028320, 5857.5903320, -1326.4201660, 3353.4309082, -5653.8339844, 7184.0102539
4: -6584.4262695, 4295.7910156, -3765.1281738, 2475.0305176, -9059.4570312, 8060.9179688

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1924999
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1871620
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -4196.5258789, 3373.1972656, -8531.5957031, 8364.0644531
1: -4135.0488281, 4011.9240723, -3375.0051270, 3246.5498047, -7381.5981445, 7386.9291992
2: -5915.9174805, 4360.9184570, -4840.1069336, 3519.6916504, -9435.6093750, 9201.0234375
3: -2300.4028320, 5857.5903320, -1852.6146240, 4779.9594727, -7080.3623047, 7710.2045898
4: -6584.4262695, 4295.7910156, -5363.3623047, 3465.0339355, -10049.4599609, 9659.1513672

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1945753
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1892375
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6424.9907227, 5166.5434570, -2952.6821289, 2402.8015137, -8827.7910156, 8119.2255859
1: -5155.3681641, 4968.0869141, -2365.3845215, 2314.5366211, -7469.9047852, 7333.4716797
2: -7383.2236328, 5399.3579102, -3383.4587402, 2517.0407715, -9900.2636719, 8782.8154297
3: -2848.6857910, 7298.5976562, -1326.4201660, 3353.4309082, -6202.1166992, 8625.0175781
4: -8197.9521484, 5322.1879883, -3765.1281738, 2475.0305176, -10672.9824219, 9087.3164062

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4646323, upper bound: 4552.1884538
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844829, upper bound: 4552.1861417
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6465.6679688, 5197.9145508, -4267.7421875, 3427.3356934, -9893.0039062, 9465.6562500
1: -5188.6606445, 4998.1103516, -3433.8293457, 3298.9086914, -8487.5683594, 8431.9394531
2: -7431.6689453, 5431.6601562, -4925.3828125, 3575.4016113, -11007.0703125, 10357.0410156
3: -2865.7429199, 7345.2402344, -1881.0106201, 4863.0942383, -7724.6611328, 9226.2509766
4: -8250.3779297, 5354.0131836, -5456.5566406, 3519.1264648, -11769.5009766, 10810.5673828

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1898073
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1872713
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -5158.3984375, 4167.5395508, -9325.9375000, 9325.9375000
1: -4135.0488281, 4011.9240723, -4135.0488281, 4011.9240723, -8146.9726562, 8146.9726562
2: -5915.9174805, 4360.9184570, -5915.9174805, 4360.9184570, -10274.1767578, 10274.1767578
3: -2300.4028320, 5857.5903320, -2300.4028320, 5857.5903320, -8157.9863281, 8157.9863281
4: -6584.4262695, 4295.7910156, -6584.4262695, 4295.7910156, -10875.1572266, 10875.1562500

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1922033
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1868654
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -6423.0395508, 5165.0283203, -10319.8144531, 10590.5791016
1: -4135.0488281, 4011.9240723, -5153.7744141, 4966.6381836, -9100.3583984, 9165.6982422
2: -5915.9174805, 4360.9184570, -7380.9018555, 5397.7973633, -11304.4179688, 11733.6875000
3: -2300.4028320, 5857.5903320, -2847.8613281, 7296.3554688, -9576.7294922, 8701.9521484
4: -6584.4262695, 4295.7910156, -8195.4394531, 5320.6503906, -11892.4355469, 12486.1279297

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1945753
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1892375
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6423.0395508, 5165.0283203, -5158.3984375, 4167.5395508, -10590.5791016, 10319.8144531
1: -5153.7744141, 4966.6381836, -4135.0488281, 4011.9240723, -9165.6982422, 9100.3583984
2: -7380.9018555, 5397.7973633, -5915.9174805, 4360.9184570, -11733.6875000, 11304.4189453
3: -2847.8613281, 7296.3554688, -2300.4028320, 5857.5903320, -8701.9531250, 9576.7294922
4: -8195.4394531, 5320.6503906, -6584.4262695, 4295.7910156, -12486.1279297, 11892.4355469

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4646323, upper bound: 4552.1819100
time: 1.07 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1795979
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6465.6679688, 5197.9145508, -6465.6679688, 5197.9145508, -11659.2441406, 11659.2441406
1: -5188.6606445, 4998.1103516, -5188.6606445, 4998.1103516, -10185.4218750, 10185.4208984
2: -7431.6689453, 5431.6601562, -7431.6689453, 5431.6601562, -12847.5341797, 12847.5341797
3: -2865.7429199, 7345.2402344, -2865.7429199, 7345.2402344, -10186.5107422, 10186.5107422
4: -8250.3779297, 5354.0131836, -8250.3779297, 5354.0131836, -13591.3330078, 13591.3330078

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1898002
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1872592
time: 0.84 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.78 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.78
Output dim: 0, lower bound: -4551.6986709, upper bound: 4551.7638377
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.78
Output dim: 0, lower bound: -4551.7920228, upper bound: 4551.7659535
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.78
Output dim: 0, lower bound: -4551.6986709, upper bound: 4551.7638377
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.78
Output dim: 0, lower bound: -4551.7920228, upper bound: 4551.7659535
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.78
Output dim: 0, lower bound: -4551.7282293, upper bound: 4551.7583069
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.78
Output dim: 0, lower bound: -4551.7536229, upper bound: 4551.7536229
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.78
Output dim: 0, lower bound: -4551.7282293, upper bound: 4551.7583069
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.78
Output dim: 0, lower bound: -4551.7536229, upper bound: 4551.7536229
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.78
Output dim: 0, lower bound: -4551.7052373, upper bound: 4551.7637713
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.78
Output dim: 0, lower bound: -4551.7985891, upper bound: 4551.7658871
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.78
Output dim: 0, lower bound: -4551.7052373, upper bound: 4551.7638109
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.78
Output dim: 0, lower bound: -4551.7985891, upper bound: 4551.7659267
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.78
Output dim: 0, lower bound: -4551.7347956, upper bound: 4551.7582405
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.78
Output dim: 0, lower bound: -4551.7601892, upper bound: 4551.7535565
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.78
Output dim: 0, lower bound: -4551.7347956, upper bound: 4551.7582713
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.78
Output dim: 0, lower bound: -4551.7601892, upper bound: 4551.7535565
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.78
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1924999
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.78
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1871620
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.78
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1945753
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.78
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1892375
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.78
Output dim: 0, lower bound: -4551.4646323, upper bound: 4552.1884538
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.78
Output dim: 0, lower bound: -4552.1844829, upper bound: 4552.1861417
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.78
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1898073
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.78
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1872713
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.78
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1922033
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.78
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1868654
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.78
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1945753
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.78
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1892375
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.78
Output dim: 0, lower bound: -4551.4646323, upper bound: 4552.1819100
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.78
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1795979
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.78
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1898002
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.78
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1872592

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -2952.6821289, 2402.8015137, -6999.9189453, 6660.2255859
1: -3687.0437012, 3569.1484375, -2365.3845215, 2314.5366211, -6001.5800781, 5934.5332031
2: -5276.2031250, 3879.7956543, -3383.4587402, 2517.0407715, -7793.2426758, 7263.2524414
3: -2047.8087158, 5221.5805664, -1326.4201660, 3353.4309082, -5401.2392578, 6547.9995117
4: -5870.9716797, 3822.0715332, -3765.1281738, 2475.0305176, -8346.0019531, 7587.1992188

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7380623, upper bound: 4551.7103340
time: 1.14 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7401350, upper bound: 4551.8036859
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -2952.6821289, 2402.8015137, -7506.7626953, 7075.9648438
1: -4091.2307129, 3969.2382812, -2365.3845215, 2314.5366211, -6405.7670898, 6334.6230469
2: -5853.0830078, 4314.3935547, -3383.4587402, 2517.0407715, -8370.1240234, 7697.8510742
3: -2275.4304199, 5795.1562500, -1326.4201660, 3353.4309082, -5628.8608398, 7121.5761719
4: -6514.6850586, 4250.0151367, -3765.1281738, 2475.0305176, -8989.7158203, 8015.1425781

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7679201, upper bound: 4551.7029061
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7699927, upper bound: 4551.7962579
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -4187.3461914, 3366.2236328, -7963.3413086, 7894.8896484
1: -3687.0437012, 3569.1484375, -3367.4365234, 3239.7897949, -6926.8334961, 6936.5849609
2: -5276.2031250, 3879.7956543, -4829.1313477, 3512.4965820, -8788.6992188, 8708.9267578
3: -2047.8087158, 5221.5805664, -1848.9407959, 4769.2617188, -6817.0698242, 7070.5200195
4: -5870.9716797, 3822.0715332, -5351.3623047, 3458.0478516, -9329.0195312, 9173.4335938

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7322538, upper bound: 4551.7398922
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7280324, upper bound: 4551.7652860
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -4186.9648438, 3365.9348145, -8469.8955078, 8310.2480469
1: -4091.2307129, 3969.2382812, -3367.1228027, 3239.5100098, -7330.7397461, 7336.3608398
2: -5853.0830078, 4314.3935547, -4828.6757812, 3512.1987305, -9365.2812500, 9143.0683594
3: -2275.4304199, 5795.1562500, -1848.7885742, 4768.8183594, -7044.2475586, 7643.9448242
4: -6514.6850586, 4250.0151367, -5350.8642578, 3457.7578125, -9972.4433594, 9600.8789062

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7621116, upper bound: 4551.7324642
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7578902, upper bound: 4551.7578580
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6324.9355469, 5087.5170898, -2952.6821289, 2402.8015137, -8727.7373047, 8040.1987305
1: -5076.7548828, 4892.7792969, -2365.3845215, 2314.5366211, -7391.2915039, 7258.1640625
2: -7273.1225586, 5315.4663086, -3383.4587402, 2517.0407715, -9790.1630859, 8698.9238281
3: -2803.7656250, 7192.3369141, -1326.4201660, 3353.4309082, -6157.1962891, 8518.7568359
4: -8073.3588867, 5238.7558594, -3765.1281738, 2475.0305176, -10548.3876953, 9003.8818359

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4292608, upper bound: 4551.6875449
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4343192, upper bound: 4551.7947080
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6349.2709961, 5105.7099609, -2952.6821289, 2402.8015137, -8752.0722656, 8058.3920898
1: -5094.4326172, 4909.3847656, -2365.3845215, 2314.5366211, -7408.9692383, 7274.7690430
2: -7296.0161133, 5335.1386719, -3383.4587402, 2517.0407715, -9813.0527344, 8718.5966797
3: -2814.6669922, 7213.2519531, -1326.4201660, 3353.4309082, -6168.0976562, 8539.6718750
4: -8101.4882812, 5259.2280273, -3765.1281738, 2475.0305176, -10576.5175781, 9024.3564453

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7427411, upper bound: 4551.6863623
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7477994, upper bound: 4551.7935255
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6366.1660156, 5119.2890625, -4267.7421875, 3427.3356934, -9793.5019531, 9387.0302734
1: -5110.7729492, 4923.2451172, -3433.8293457, 3298.9086914, -8409.6816406, 8357.0742188
2: -7322.4482422, 5348.0849609, -4925.3828125, 3575.4016113, -10897.8486328, 10273.4677734
3: -2820.9536133, 7239.8808594, -1881.0106201, 4863.0942383, -7676.4799805, 9120.8203125
4: -8126.7597656, 5270.9204102, -5456.5566406, 3519.1264648, -11645.8867188, 10727.4765625

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4241378, upper bound: 4551.7171103
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4219976, upper bound: 4551.7437677
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6390.3413086, 5137.2963867, -4267.7421875, 3427.3356934, -9817.6767578, 9405.0390625
1: -5128.1879883, 4939.5961914, -3433.8293457, 3298.9086914, -8427.0957031, 8373.4257812
2: -7344.9951172, 5367.6391602, -4925.3828125, 3575.4016113, -10920.3964844, 10293.0214844
3: -2831.8232422, 7260.3583984, -1881.0106201, 4863.0942383, -7687.5229492, 9141.3691406
4: -8154.4711914, 5291.2568359, -5456.5566406, 3519.1264648, -11673.5957031, 10747.8134766

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7376181, upper bound: 4551.7160190
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7354778, upper bound: 4551.7425851
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -5158.3984375, 4167.5395508, -8764.6572266, 8865.9423828
1: -3687.0437012, 3569.1484375, -4135.0488281, 4011.9240723, -7698.9667969, 7704.1972656
2: -5276.2031250, 3879.7956543, -5915.9174805, 4360.9184570, -9632.9892578, 9791.7519531
3: -2047.8087158, 5221.5805664, -2300.4028320, 5857.5903320, -7904.4311523, 7519.7099609
4: -5870.9716797, 3822.0715332, -6584.4262695, 4295.7910156, -10160.3740234, 10400.6279297

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1621650, upper bound: 4552.1621650
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1621650, upper bound: 4552.1916969
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -5158.3984375, 4167.5395508, -9271.5009766, 9281.6806641
1: -4091.2307129, 3969.2382812, -4135.0488281, 4011.9240723, -8103.1542969, 8104.2871094
2: -5853.0830078, 4314.3935547, -5915.9174805, 4360.9184570, -10211.1474609, 10227.9863281
3: -2275.4304199, 5795.1562500, -2300.4028320, 5857.5903320, -8133.0200195, 8094.7338867
4: -6514.6850586, 4250.0151367, -6584.4262695, 4295.7910156, -10805.3154297, 10829.7656250

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1916969, upper bound: 4552.1621650
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1916969, upper bound: 4552.1916969
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -6416.0761719, 5159.6215820, -9751.5517578, 10123.6201172
1: -3687.0437012, 3569.1484375, -5148.1215820, 4961.4672852, -8646.2851562, 8717.2685547
2: -5276.2031250, 3879.7956543, -7372.6123047, 5392.2304688, -10657.6738281, 11243.1250000
3: -2047.8087158, 5221.5805664, -2844.9218750, 7288.3549805, -9315.3183594, 8060.7441406
4: -5870.9716797, 3822.0715332, -8186.4633789, 5315.1650391, -11172.1748047, 12002.6777344

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1545275, upper bound: 4551.4742097
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1513498, upper bound: 4552.1940604
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -6415.7875977, 5159.3974609, -10259.6679688, 10539.0693359
1: -4091.2307129, 3969.2382812, -5147.8876953, 4961.2529297, -9051.0683594, 9117.1259766
2: -5853.0830078, 4314.3935547, -7372.2680664, 5391.9985352, -11235.6035156, 11679.0234375
3: -2275.4304199, 5795.1562500, -2844.8002930, 7288.0244141, -9543.7695312, 8635.6455078
4: -6514.6850586, 4250.0151367, -8186.0927734, 5314.9379883, -11816.8886719, 12431.4472656

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1794140, upper bound: 4551.4667226
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1762363, upper bound: 4552.1865733
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6322.9863281, 5086.0170898, -5158.3984375, 4167.5395508, -10490.3603516, 10237.9902344
1: -5075.1489258, 4891.3442383, -4135.0488281, 4011.9240723, -9087.0732422, 9022.2626953
2: -7270.7949219, 5313.9262695, -5915.9174805, 4360.9184570, -11620.4501953, 11216.8486328
3: -2802.9521484, 7190.0961914, -2300.4028320, 5857.5903320, -8653.6865234, 9469.9433594
4: -8070.8398438, 5237.2363281, -6584.4262695, 4295.7910156, -12358.8896484, 11805.5585938

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4742097, upper bound: 4552.1554065
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4667226, upper bound: 4552.1802930
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6347.3222656, 5104.2104492, -5158.3984375, 4167.5395508, -10514.8613281, 10255.9130859
1: -5092.8393555, 4907.9492188, -4135.0488281, 4011.9240723, -9104.7636719, 9038.9326172
2: -7293.6977539, 5333.5937500, -5915.9174805, 4360.9184570, -11647.3525391, 11236.6035156
3: -2813.8508301, 7211.0209961, -2300.4028320, 5857.5903320, -8664.7587891, 9491.3281250
4: -8098.9799805, 5257.7055664, -6584.4262695, 4295.7910156, -12390.7763672, 11825.7529297

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1940604, upper bound: 4552.1529031
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1865733, upper bound: 4552.1777897
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6366.1660156, 5119.2890625, -6465.6679688, 5197.9145508, -11557.8154297, 11577.7519531
1: -5110.7729492, 4923.2451172, -5188.6606445, 4998.1103516, -10105.2763672, 10107.7431641
2: -7322.4482422, 5348.0849609, -7431.6689453, 5431.6601562, -12735.1406250, 12760.1982422
3: -2820.9536133, 7239.8808594, -2865.7429199, 7345.2402344, -10138.3300781, 10080.5986328
4: -8126.7597656, 5270.9204102, -8250.3779297, 5354.0131836, -13465.0605469, 13504.7070312

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4646321, upper bound: 4551.4668377
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1872591
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6390.3413086, 5137.2963867, -6465.6679688, 5197.9145508, -11584.8183594, 11595.4892578
1: -5128.1879883, 4939.5961914, -5188.6606445, 4998.1103516, -10125.4355469, 10124.1347656
2: -7344.9951172, 5367.6391602, -7431.6689453, 5431.6601562, -12761.6904297, 12779.8447266
3: -2831.8232422, 7260.3583984, -2865.7429199, 7345.2402344, -10149.3730469, 10101.5341797
4: -8154.4711914, 5291.2568359, -8250.3779297, 5354.0131836, -13496.5107422, 13524.7314453

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844829, upper bound: 4551.4668377
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844829, upper bound: 4552.1872591
time: 0.87 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.78 seconds
IS_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.78
Output dim: 0, lower bound: -4551.7380623, upper bound: 4551.7103340
IS_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.78
Output dim: 0, lower bound: -4551.7401350, upper bound: 4551.8036859
IS_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.78
Output dim: 0, lower bound: -4551.7679201, upper bound: 4551.7029061
IS_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.78
Output dim: 0, lower bound: -4551.7699927, upper bound: 4551.7962579
IS_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.78
Output dim: 0, lower bound: -4551.7322538, upper bound: 4551.7398922
IS_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.78
Output dim: 0, lower bound: -4551.7280324, upper bound: 4551.7652860
IS_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.78
Output dim: 0, lower bound: -4551.7621116, upper bound: 4551.7324642
IS_A2_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.78
Output dim: 0, lower bound: -4551.7578902, upper bound: 4551.7578580
IS_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.78
Output dim: 0, lower bound: -4550.4292608, upper bound: 4551.6875449
IS_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.78
Output dim: 0, lower bound: -4550.4343192, upper bound: 4551.7947080
IS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.78
Output dim: 0, lower bound: -4551.7427411, upper bound: 4551.6863623
IS_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.78
Output dim: 0, lower bound: -4551.7477994, upper bound: 4551.7935255
IS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.78
Output dim: 0, lower bound: -4550.4241378, upper bound: 4551.7171103
IS_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.78
Output dim: 0, lower bound: -4550.4219976, upper bound: 4551.7437677
IS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.78
Output dim: 0, lower bound: -4551.7376181, upper bound: 4551.7160190
IS_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.78
Output dim: 0, lower bound: -4551.7354778, upper bound: 4551.7425851
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -4552.1621650, upper bound: 4552.1621650
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -4552.1621650, upper bound: 4552.1916969
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -4552.1916969, upper bound: 4552.1621650
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -4552.1916969, upper bound: 4552.1916969
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -4552.1545275, upper bound: 4551.4742097
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -4552.1513498, upper bound: 4552.1940604
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -4552.1794140, upper bound: 4551.4667226
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -4552.1762363, upper bound: 4552.1865733
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -4551.4742097, upper bound: 4552.1554065
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -4551.4667226, upper bound: 4552.1802930
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -4552.1940604, upper bound: 4552.1529031
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -4552.1865733, upper bound: 4552.1777897
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.78
Output dim: 0, lower bound: -4551.4646321, upper bound: 4551.4668377
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1872591
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -4552.1844829, upper bound: 4551.4668377
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -4552.1844829, upper bound: 4552.1872591

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -4597.1176758, 3707.5437012, -8304.6611328, 8304.6611328
1: -3687.0437012, 3569.1484375, -3687.0437012, 3569.1484375, -7256.1918945, 7256.1918945
2: -5276.2031250, 3879.7956543, -5276.2031250, 3879.7956543, -9150.5644531, 9150.5644531
3: -2047.8087158, 5221.5805664, -2047.8087158, 5221.5805664, -7266.1542969, 7266.1542969
4: -5870.9716797, 3822.0715332, -5870.9716797, 3822.0715332, -9685.8447266, 9685.8437500

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.7139066, upper bound: 4551.7342625
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7323025, upper bound: 4551.7397307
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -5103.9614258, 4123.2832031, -8720.4003906, 8811.5039062
1: -3687.0437012, 3569.1484375, -4091.2307129, 3969.2382812, -7656.2822266, 7660.3789062
2: -5276.2031250, 3879.7956543, -5853.0830078, 4314.3935547, -9586.7988281, 9728.7226562
3: -2047.8087158, 5221.5805664, -2275.4304199, 5795.1562500, -7841.1791992, 7494.9267578
4: -5870.9716797, 3822.0715332, -6514.6850586, 4250.0151367, -10114.9833984, 10330.7871094

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.7139066, upper bound: 4551.7641203
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7323025, upper bound: 4551.7695884
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -4597.1176758, 3707.5437012, -8811.5029297, 8720.4003906
1: -4091.2307129, 3969.2382812, -3687.0437012, 3569.1484375, -7660.3789062, 7656.2822266
2: -5853.0830078, 4314.3935547, -5276.2031250, 3879.7956543, -9728.7226562, 9586.7988281
3: -2275.4304199, 5795.1562500, -2047.8087158, 5221.5805664, -7494.9267578, 7841.1787109
4: -6514.6850586, 4250.0151367, -5870.9716797, 3822.0715332, -10330.7861328, 10114.9833984

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6117698, upper bound: 4551.7330142
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7621604, upper bound: 4551.7323027
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -5103.9614258, 4123.2832031, -9227.2421875, 9227.2431641
1: -4091.2307129, 3969.2382812, -4091.2307129, 3969.2382812, -8060.4687500, 8060.4687500
2: -5853.0830078, 4314.3935547, -5853.0830078, 4314.3935547, -10164.9560547, 10164.9570312
3: -2275.4304199, 5795.1562500, -2275.4304199, 5795.1562500, -8069.9511719, 8069.9506836
4: -6514.6850586, 4250.0151367, -6514.6850586, 4250.0151367, -10759.9257812, 10759.9248047

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6117698, upper bound: 4551.7623046
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7621604, upper bound: 4551.7590118
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -6316.0258789, 5080.6542969, -9669.7832031, 10022.6650391
1: -3687.0437012, 3569.1484375, -5069.4101562, 4886.2128906, -8568.2314453, 8638.5585938
2: -5276.2031250, 3879.7956543, -7262.4819336, 5308.4174805, -10570.1738281, 11129.8652344
3: -2047.8087158, 5221.5805664, -2800.0471191, 7182.0898438, -9208.5312500, 8012.5190430
4: -5870.9716797, 3822.0715332, -8061.8393555, 5231.8027344, -11085.3623047, 11875.4140625

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4445115, upper bound: 4550.4219729
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4968185, upper bound: 4550.4290304
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -6340.3676758, 5098.8491211, -9687.7080078, 10047.9111328
1: -3687.0437012, 3569.1484375, -5087.1557617, 4902.8212891, -8584.9082031, 8656.3046875
2: -5276.2031250, 3879.7956543, -7285.4267578, 5328.0751953, -10589.9169922, 11156.8115234
3: -2047.8087158, 5221.5805664, -2810.9357910, 7203.0576172, -9229.9570312, 8023.5805664
4: -5870.9716797, 3822.0715332, -8090.0302734, 5252.2690430, -11105.5576172, 11907.3525391

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4444965, upper bound: 4551.7388089
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7091438, upper bound: 4551.7470782
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -6315.7363281, 5080.4316406, -10177.9013672, 10439.0195312
1: -4091.2307129, 3969.2382812, -5069.1728516, 4885.9990234, -8973.0156250, 9038.4111328
2: -5853.0830078, 4314.3935547, -7262.1372070, 5308.1889648, -11148.1054688, 11565.7617188
3: -2275.4304199, 5795.1562500, -2799.9265137, 7181.7592773, -9436.9785156, 8587.4228516
4: -6514.6850586, 4250.0151367, -8061.4667969, 5231.5781250, -11730.0820312, 12304.1806641

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.1823498, upper bound: 4550.4219544
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4984060, upper bound: 4550.4219561
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -6340.0795898, 5098.6284180, -10195.8271484, 10463.3623047
1: -4091.2307129, 3969.2382812, -5086.9204102, 4902.6079102, -8989.6923828, 9056.1582031
2: -5853.0830078, 4314.3935547, -7285.0854492, 5327.8457031, -11167.8496094, 11592.7089844
3: -2275.4304199, 5795.1562500, -2810.8146973, 7202.7280273, -9458.4091797, 8598.4843750
4: -6514.6850586, 4250.0151367, -8089.6596680, 5252.0434570, -11750.2753906, 12336.1230469

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.5933311, upper bound: 4551.7388911
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7376065, upper bound: 4551.7391450
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6316.0258789, 5080.6542969, -4597.1176758, 3707.5437012, -10022.6650391, 9669.7822266
1: -5069.4101562, 4886.2128906, -3687.0437012, 3569.1484375, -8638.5585938, 8568.2314453
2: -7262.4819336, 5308.4174805, -5276.2031250, 3879.7956543, -11129.8652344, 10570.1738281
3: -2800.0471191, 7182.0898438, -2047.8087158, 5221.5805664, -8012.5185547, 9208.5312500
4: -8061.8393555, 5231.8027344, -5870.9716797, 3822.0715332, -11875.4130859, 11085.3623047

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.8832384, upper bound: 4551.7139337
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4302495, upper bound: 4551.7125020
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6315.7363281, 5080.4316406, -5103.9614258, 4123.2832031, -10439.0195312, 10177.9013672
1: -5069.1728516, 4885.9990234, -4091.2307129, 3969.2382812, -9038.4111328, 8973.0156250
2: -7262.1372070, 5308.1889648, -5853.0830078, 4314.3935547, -11565.7617188, 11148.1064453
3: -2799.9265137, 7181.7592773, -2275.4304199, 5795.1562500, -8587.4228516, 9436.9785156
4: -8061.4667969, 5231.5781250, -6514.6850586, 4250.0151367, -12304.1806641, 11730.0820312

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.8753052, upper bound: 4551.7423962
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4223163, upper bound: 4551.7409645
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6340.3676758, 5098.8491211, -4597.1176758, 3707.5437012, -10047.9111328, 9687.7080078
1: -5087.1557617, 4902.8212891, -3687.0437012, 3569.1484375, -8656.3046875, 8584.9082031
2: -7285.4267578, 5328.0751953, -5276.2031250, 3879.7956543, -11156.8115234, 10589.9179688
3: -2810.9357910, 7203.0576172, -2047.8087158, 5221.5805664, -8023.5800781, 9229.9580078
4: -8090.0302734, 5252.2690430, -5870.9716797, 3822.0715332, -11907.3535156, 11105.5585938

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6049666, upper bound: 4551.7124846
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7470782, upper bound: 4551.7124870
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6340.0795898, 5098.6284180, -5103.9614258, 4123.2832031, -10463.3623047, 10195.8271484
1: -5086.9204102, 4902.6079102, -4091.2307129, 3969.2382812, -9056.1582031, 8989.6923828
2: -7285.0854492, 5327.8457031, -5853.0830078, 4314.3935547, -11592.7099609, 11167.8486328
3: -2810.8146973, 7202.7280273, -2275.4304199, 5795.1562500, -8598.4843750, 9458.4091797
4: -8089.6596680, 5252.0434570, -6514.6850586, 4250.0151367, -12336.1230469, 11750.2753906

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7149680, upper bound: 4551.7409471
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7391450, upper bound: 4551.7409495
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6366.1660156, 5119.2890625, -6390.3413086, 5137.2963867, -11494.0605469, 11503.3271484
1: -5110.7729492, 4923.2451172, -5128.1879883, 4939.5961914, -10043.9892578, 10047.7578125
2: -7322.4482422, 5348.0849609, -7344.9951172, 5367.6391602, -12667.4521484, 12674.3535156
3: -2820.9536133, 7239.8808594, -2831.8232422, 7260.3583984, -10053.3544922, 10043.4609375
4: -8126.7597656, 5270.9204102, -8154.4711914, 5291.2568359, -13398.4580078, 13409.8847656

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.7561070, upper bound: 4551.7420811
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4258744, upper bound: 4551.7417620
time: 1.15 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6390.3413086, 5137.2963867, -6366.1660156, 5119.2890625, -11503.3271484, 11494.0615234
1: -5128.1879883, 4939.5961914, -5110.7729492, 4923.2451172, -10047.7568359, 10043.9892578
2: -7344.9951172, 5367.6391602, -7322.4482422, 5348.0849609, -12674.3544922, 12667.4521484
3: -2831.8232422, 7260.3583984, -2820.9536133, 7239.8808594, -10043.4609375, 10053.3535156
4: -8154.4711914, 5291.2568359, -8126.7597656, 5270.9204102, -13409.8847656, 13398.4580078

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.1828972, upper bound: 4550.4258699
time: 0.96 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.5417017, upper bound: 4550.4273814
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6390.3413086, 5137.2963867, -6390.3413086, 5137.2963867, -11521.0634766, 11521.0644531
1: -5128.1879883, 4939.5961914, -5128.1879883, 4939.5961914, -10064.1494141, 10064.1494141
2: -7344.9951172, 5367.6391602, -7344.9951172, 5367.6391602, -12694.0019531, 12694.0019531
3: -2831.8232422, 7260.3583984, -2831.8232422, 7260.3583984, -10064.3964844, 10064.3964844
4: -8154.4711914, 5291.2568359, -8154.4711914, 5291.2568359, -13429.9082031, 13429.9091797

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.5549223, upper bound: 4551.7333355
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.5417018, upper bound: 4551.7375208
time: 0.87 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.72 seconds
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.72
Output dim: 0, lower bound: -4550.7139066, upper bound: 4551.7342625
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.72
Output dim: 0, lower bound: -4551.7323025, upper bound: 4551.7397307
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.72
Output dim: 0, lower bound: -4550.7139066, upper bound: 4551.7641203
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.72
Output dim: 0, lower bound: -4551.7323025, upper bound: 4551.7695884
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.72
Output dim: 0, lower bound: -4551.6117698, upper bound: 4551.7330142
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.72
Output dim: 0, lower bound: -4551.7621604, upper bound: 4551.7323027
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.72
Output dim: 0, lower bound: -4551.6117698, upper bound: 4551.7623046
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.72
Output dim: 0, lower bound: -4551.7621604, upper bound: 4551.7590118
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.72
Output dim: 0, lower bound: -4550.4445115, upper bound: 4550.4219729
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.72
Output dim: 0, lower bound: -4551.4968185, upper bound: 4550.4290304
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.72
Output dim: 0, lower bound: -4550.4444965, upper bound: 4551.7388089
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.72
Output dim: 0, lower bound: -4551.7091438, upper bound: 4551.7470782
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.72
Output dim: 0, lower bound: -4551.1823498, upper bound: 4550.4219544
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.72
Output dim: 0, lower bound: -4551.4984060, upper bound: 4550.4219561
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.72
Output dim: 0, lower bound: -4551.5933311, upper bound: 4551.7388911
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.72
Output dim: 0, lower bound: -4551.7376065, upper bound: 4551.7391450
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.72
Output dim: 0, lower bound: -4550.8832384, upper bound: 4551.7139337
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.72
Output dim: 0, lower bound: -4550.4302495, upper bound: 4551.7125020
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.72
Output dim: 0, lower bound: -4550.8753052, upper bound: 4551.7423962
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.72
Output dim: 0, lower bound: -4550.4223163, upper bound: 4551.7409645
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.72
Output dim: 0, lower bound: -4551.6049666, upper bound: 4551.7124846
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.72
Output dim: 0, lower bound: -4551.7470782, upper bound: 4551.7124870
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.72
Output dim: 0, lower bound: -4551.7149680, upper bound: 4551.7409471
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.72
Output dim: 0, lower bound: -4551.7391450, upper bound: 4551.7409495
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.72
Output dim: 0, lower bound: -4550.7561070, upper bound: 4551.7420811
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.72
Output dim: 0, lower bound: -4550.4258744, upper bound: 4551.7417620
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.72
Output dim: 0, lower bound: -4551.1828972, upper bound: 4550.4258699
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.72
Output dim: 0, lower bound: -4551.5417017, upper bound: 4550.4273814
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.72
Output dim: 0, lower bound: -4551.5549223, upper bound: 4551.7333355
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.72
Output dim: 0, lower bound: -4551.5417018, upper bound: 4551.7375208
Binary search (step 2): status=Status.VERIFIED, low=0.8750000, high=1.0000000, mid=0.8750000, abs_max=5687.5751953125
rel_dist={0: [-4552.289001211694, 4552.289001211693]}

## Binary search (step 3) starts
Candidate diff: 0.9375000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2865166, upper bound: 4552.2783404
time: 0.80 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711
time: 1.13 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.11 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.11
Output dim: 0, lower bound: -4552.2865166, upper bound: 4552.2783404
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.11
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -2987.9899902, 2430.9243164, -3136.6748047, 2550.9006348, -5538.8906250, 5567.5991211
1: -2393.8061523, 2341.6025391, -2513.2941895, 2457.4460449, -4851.2519531, 4854.8964844
2: -3424.2626953, 2546.4648438, -3595.4755859, 2672.1398926, -6096.4023438, 6141.9404297
3: -1341.7878418, 3393.8977051, -1407.5072021, 3564.2558594, -4906.0439453, 4801.4047852
4: -3810.2302246, 2503.8750000, -3999.4067383, 2627.0041504, -6437.2343750, 6503.2817383

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711
time: 0.98 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711
time: 0.81 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -5188.6577148, 4191.8315430, -3136.6748047, 2550.9006348, -7739.5585938, 7328.5063477
1: -4159.4165039, 4035.3022461, -2513.2941895, 2457.4460449, -6616.8623047, 6548.5966797
2: -5950.7929688, 4386.3300781, -3595.4755859, 2672.1398926, -8622.9326172, 7981.8056641
3: -2313.8149414, 5892.1474609, -1407.5072021, 3564.2558594, -5878.0708008, 7299.6542969
4: -6623.0805664, 4320.7236328, -3999.4067383, 2627.0041504, -9250.0830078, 8320.1308594

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711
time: 0.86 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711
time: 0.74 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.78 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.78
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.78
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.78
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.78
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -2987.9899902, 2430.9243164, -2987.9899902, 2430.9243164, -5418.9140625, 5418.9140625
1: -2393.8061523, 2341.6025391, -2393.8061523, 2341.6025391, -4735.4086914, 4735.4086914
2: -3424.2626953, 2546.4648438, -3424.2626953, 2546.4648438, -5970.7275391, 5970.7275391
3: -1341.7878418, 3393.8977051, -1341.7878418, 3393.8977051, -4735.6855469, 4735.6855469
4: -3810.2302246, 2503.8750000, -3810.2302246, 2503.8750000, -6314.1054688, 6314.1054688

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2085840, upper bound: 4552.1951258
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1880421
time: 0.74 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -2987.9899902, 2430.9243164, -5188.6577148, 4191.8315430, -7179.8212891, 7619.5820312
1: -2393.8061523, 2341.6025391, -4159.4165039, 4035.3022461, -6429.1083984, 6501.0190430
2: -3424.2626953, 2546.4648438, -5950.7929688, 4386.3300781, -7810.5922852, 8497.2558594
3: -1341.7878418, 3393.8977051, -2313.8149414, 5892.1474609, -7233.9355469, 5707.7128906
4: -3810.2302246, 2503.8750000, -6623.0805664, 4320.7236328, -8130.9526367, 9126.9550781

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2085840, upper bound: 4552.1951259
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1880421
time: 0.85 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -5188.6577148, 4191.8315430, -2987.9899902, 2430.9243164, -7619.5820312, 7179.8212891
1: -4159.4165039, 4035.3022461, -2393.8061523, 2341.6025391, -6501.0190430, 6429.1083984
2: -5950.7929688, 4386.3300781, -3424.2626953, 2546.4648438, -8497.2558594, 7810.5922852
3: -2313.8149414, 5892.1474609, -1341.7878418, 3393.8977051, -5707.7128906, 7233.9355469
4: -6623.0805664, 4320.7236328, -3810.2302246, 2503.8750000, -9126.9550781, 8130.9531250

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1946649
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1905380
time: 1.02 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -5188.6577148, 4191.8315430, -5188.6577148, 4191.8315430, -9380.4892578, 9380.4892578
1: -4159.4165039, 4035.3022461, -4159.4165039, 4035.3022461, -8194.7187500, 8194.7187500
2: -5950.7929688, 4386.3300781, -5950.7929688, 4386.3300781, -10334.5771484, 10334.5771484
3: -2313.8149414, 5892.1474609, -2313.8149414, 5892.1474609, -8205.9628906, 8205.9628906
4: -6623.0805664, 4320.7236328, -6623.0805664, 4320.7236328, -10938.7910156, 10938.7919922

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1946649
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1905380
time: 0.79 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.44 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 0, lower bound: -4552.2085840, upper bound: 4552.1951258
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1880421
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 0, lower bound: -4552.2085840, upper bound: 4552.1951259
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1880421
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1946649
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1905380
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1946649
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1905380

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -2987.9899902, 2430.9243164, -5383.6064453, 5390.7915039
1: -2365.3845215, 2314.5366211, -2393.8061523, 2341.6025391, -4706.9873047, 4708.3427734
2: -3383.4587402, 2517.0407715, -3424.2626953, 2546.4648438, -5929.9228516, 5941.3027344
3: -1326.4201660, 3353.4309082, -1341.7878418, 3393.8977051, -4720.3173828, 4695.2187500
4: -3765.1281738, 2475.0305176, -3810.2302246, 2503.8750000, -6269.0029297, 6285.2602539

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1859666, upper bound: 4552.1859666
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1859666, upper bound: 4552.1880421
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4215.5048828, 3387.6472168, -2987.9899902, 2430.9243164, -6646.4291992, 6375.6372070
1: -3390.6635742, 3260.5351562, -2393.8061523, 2341.6025391, -5732.2661133, 5654.3413086
2: -4862.8227539, 3534.5725098, -3424.2626953, 2546.4648438, -7409.2875977, 6958.8349609
3: -1860.2094727, 4802.1191406, -1341.7878418, 3393.8977051, -5254.1074219, 6143.9072266
4: -5388.1943359, 3479.4816895, -3810.2302246, 2503.8750000, -7892.0693359, 7289.7114258

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1859666
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1880421
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -5188.6577148, 4191.8315430, -7144.5136719, 7591.4589844
1: -2365.3845215, 2314.5366211, -4159.4165039, 4035.3022461, -6400.6865234, 6473.9531250
2: -3383.4587402, 2517.0407715, -5950.7929688, 4386.3300781, -7769.7875977, 8467.8320312
3: -1326.4201660, 3353.4309082, -2313.8149414, 5892.1474609, -7218.5668945, 5667.2460938
4: -3765.1281738, 2475.0305176, -6623.0805664, 4320.7236328, -8085.8510742, 9098.1093750

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1884626, upper bound: 4552.1856700
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1884626, upper bound: 4552.1880421
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4213.3066406, 3385.9702148, -5188.6577148, 4191.8315430, -8405.1376953, 8574.6269531
1: -3388.8491211, 3258.9143066, -4159.4165039, 4035.3022461, -7424.1513672, 7418.3310547
2: -4860.1904297, 3532.8481445, -5950.7929688, 4386.3300781, -9246.5205078, 9483.6406250
3: -1859.3298340, 4799.5498047, -2313.8149414, 5892.1474609, -7751.4765625, 7113.3647461
4: -5385.3159180, 3477.8078613, -6623.0805664, 4320.7236328, -9706.0390625, 10100.8876953

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905380, upper bound: 4552.1856700
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905380, upper bound: 4552.1880421
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -2987.9899902, 2430.9243164, -7589.3227539, 7155.5288086
1: -4135.0488281, 4011.9240723, -2393.8061523, 2341.6025391, -6476.6513672, 6405.7290039
2: -5915.9174805, 4360.9184570, -3424.2626953, 2546.4648438, -8462.3818359, 7785.1806641
3: -2300.4028320, 5857.5903320, -1341.7878418, 3393.8977051, -5694.3007812, 7199.3779297
4: -6584.4262695, 4295.7910156, -3810.2302246, 2503.8750000, -9088.3007812, 8106.0195312

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1884626
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1905380
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6437.5600586, 5176.2553711, -2987.9899902, 2430.9243164, -8868.4843750, 8164.2451172
1: -5165.6396484, 4977.3759766, -2393.8061523, 2341.6025391, -7507.2421875, 7371.1816406
2: -7398.1796875, 5409.3603516, -3424.2626953, 2546.4648438, -9944.6445312, 8833.6201172
3: -2853.9689941, 7313.0058594, -1341.7878418, 3393.8977051, -6247.8662109, 8654.7939453
4: -8214.1406250, 5332.0424805, -3810.2302246, 2503.8750000, -10718.0146484, 9142.2724609

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1884626
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1905380
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -5188.6577148, 4191.8315430, -9350.2294922, 9356.1972656
1: -4135.0488281, 4011.9240723, -4159.4165039, 4035.3022461, -8170.3510742, 8171.3408203
2: -5915.9174805, 4360.9184570, -5950.7929688, 4386.3300781, -10299.6093750, 10309.1445312
3: -2300.4028320, 5857.5903320, -2313.8149414, 5892.1474609, -8192.5507812, 8171.4052734
4: -6584.4262695, 4295.7910156, -6623.0805664, 4320.7236328, -10900.1123047, 10913.8359375

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1881660
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1905380
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6435.8691406, 5174.9516602, -5188.6577148, 4191.8315430, -10627.7001953, 10360.0019531
1: -5164.2563477, 4976.1289062, -4159.4165039, 4035.3022461, -9199.5576172, 9134.2187500
2: -7396.1674805, 5408.0161133, -5950.7929688, 4386.3300781, -11774.1015625, 11349.5830078
3: -2853.2595215, 7311.0693359, -2313.8149414, 5892.1474609, -8742.0595703, 9604.6162109
4: -8211.9619141, 5330.7177734, -6623.0805664, 4320.7236328, -12527.5107422, 11941.1699219

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1881660
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1905380
time: 0.83 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.24 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -4552.1859666, upper bound: 4552.1859666
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -4552.1859666, upper bound: 4552.1880421
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1859666
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1880421
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -4552.1884626, upper bound: 4552.1856700
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -4552.1884626, upper bound: 4552.1880421
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -4552.1905380, upper bound: 4552.1856700
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -4552.1905380, upper bound: 4552.1880421
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1884626
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1905380
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1884626
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1905380
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1881660
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1905380
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1881660
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1905380

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -2952.6821289, 2402.8015137, -5355.4833984, 5355.4833984
1: -2365.3845215, 2314.5366211, -2365.3845215, 2314.5366211, -4679.9208984, 4679.9208984
2: -3383.4587402, 2517.0407715, -3383.4587402, 2517.0407715, -5900.4965820, 5900.4970703
3: -1326.4201660, 3353.4309082, -1326.4201660, 3353.4309082, -4679.8505859, 4679.8505859
4: -3765.1281738, 2475.0305176, -3765.1281738, 2475.0305176, -6240.1582031, 6240.1582031

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6986709, upper bound: 4551.7638377
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7920228, upper bound: 4551.7659535
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -4200.0634766, 3375.8857422, -6328.5668945, 6602.8652344
1: -2365.3845215, 2314.5366211, -3377.9233398, 3249.1557617, -5614.5400391, 5692.4599609
2: -3383.4587402, 2517.0407715, -4844.3393555, 3522.4648438, -6905.9223633, 7361.3798828
3: -1326.4201660, 3353.4309082, -1854.0303955, 4784.0844727, -6110.5043945, 5207.4614258
4: -3765.1281738, 2475.0305176, -5367.9897461, 3467.7263184, -7232.8540039, 7843.0205078

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6986709, upper bound: 4551.7638377
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7920228, upper bound: 4551.7638377
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4200.0634766, 3375.8857422, -2952.6821289, 2402.8015137, -6602.8652344, 6328.5668945
1: -3377.9233398, 3249.1557617, -2365.3845215, 2314.5366211, -5692.4599609, 5614.5400391
2: -4844.3393555, 3522.4648438, -3383.4587402, 2517.0407715, -7361.3789062, 6905.9228516
3: -1854.0303955, 4784.0844727, -1326.4201660, 3353.4309082, -5207.4614258, 6110.5043945
4: -5367.9897461, 3467.7263184, -3765.1281738, 2475.0305176, -7843.0200195, 7232.8540039

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7282293, upper bound: 4551.7583069
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7536229, upper bound: 4551.7536229
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4267.7421875, 3427.3356934, -4267.7421875, 3427.3356934, -7695.0771484, 7695.0771484
1: -3433.8293457, 3298.9086914, -3433.8293457, 3298.9086914, -6732.7377930, 6732.7377930
2: -4925.3828125, 3575.4016113, -4925.3828125, 3575.4016113, -8500.7832031, 8500.7841797
3: -1881.0106201, 4863.0942383, -1881.0106201, 4863.0942383, -6744.1049805, 6744.1049805
4: -5456.5566406, 3519.1264648, -5456.5566406, 3519.1264648, -8975.6835938, 8975.6835938

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7282293, upper bound: 4551.7583069
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7536229, upper bound: 4551.7536229
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -5158.3984375, 4167.5395508, -7120.2216797, 7561.2001953
1: -2365.3845215, 2314.5366211, -4135.0488281, 4011.9240723, -6377.3085938, 6449.5854492
2: -3383.4587402, 2517.0407715, -5915.9174805, 4360.9184570, -7744.3754883, 8432.9580078
3: -1326.4201660, 3353.4309082, -2300.4028320, 5857.5903320, -7184.0102539, 5653.8339844
4: -3765.1281738, 2475.0305176, -6584.4262695, 4295.7910156, -8060.9179688, 9059.4570312

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6986709, upper bound: 4551.7637713
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7985891, upper bound: 4551.7658871
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -6425.7265625, 5167.1157227, -8119.7968750, 8828.5283203
1: -2365.3845215, 2314.5366211, -5155.9707031, 4968.6342773, -7334.0185547, 7470.5073242
2: -3383.4587402, 2517.0407715, -7384.1015625, 5399.9472656, -8783.4042969, 9901.1425781
3: -1326.4201660, 3353.4309082, -2848.9968262, 7299.4448242, -8625.8652344, 6202.4277344
4: -3765.1281738, 2475.0305176, -8198.9033203, 5322.7695312, -9087.8964844, 10673.9316406

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6986709, upper bound: 4551.7638109
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7985891, upper bound: 4551.7659267
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4197.5185547, 3373.9511719, -5158.3984375, 4167.5395508, -8365.0585938, 8532.3496094
1: -3375.8239746, 3247.2807617, -4135.0488281, 4011.9240723, -7387.7480469, 7382.3295898
2: -4841.2929688, 3520.4689941, -5915.9174805, 4360.9184570, -9202.2109375, 9436.3867188
3: -1853.0117188, 4781.1162109, -2300.4028320, 5857.5903320, -7710.6020508, 7081.5190430
4: -5364.6601562, 3465.7893066, -6584.4262695, 4295.7910156, -9660.4462891, 10050.2158203

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7347956, upper bound: 4551.7582405
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7601892, upper bound: 4551.7535565
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4267.7421875, 3427.3356934, -6465.6679688, 5197.9145508, -9465.6562500, 9893.0039062
1: -3433.8293457, 3298.9086914, -5188.6606445, 4998.1103516, -8431.9394531, 8487.5683594
2: -4925.3828125, 3575.4016113, -7431.6689453, 5431.6601562, -10357.0410156, 11007.0703125
3: -1881.0106201, 4863.0942383, -2865.7429199, 7345.2402344, -9226.2509766, 7724.6606445
4: -5456.5566406, 3519.1264648, -8250.3779297, 5354.0131836, -10810.5673828, 11769.5009766

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7347956, upper bound: 4551.7582713
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7601892, upper bound: 4551.7535565
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -2952.6821289, 2402.8015137, -7561.2001953, 7120.2216797
1: -4135.0488281, 4011.9240723, -2365.3845215, 2314.5366211, -6449.5854492, 6377.3085938
2: -5915.9174805, 4360.9184570, -3383.4587402, 2517.0407715, -8432.9580078, 7744.3750000
3: -2300.4028320, 5857.5903320, -1326.4201660, 3353.4309082, -5653.8339844, 7184.0102539
4: -6584.4262695, 4295.7910156, -3765.1281738, 2475.0305176, -9059.4570312, 8060.9179688

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1924999
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1871620
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -4197.5185547, 3373.9511719, -8532.3496094, 8365.0576172
1: -4135.0488281, 4011.9240723, -3375.8239746, 3247.2807617, -7382.3295898, 7387.7480469
2: -5915.9174805, 4360.9184570, -4841.2929688, 3520.4689941, -9436.3867188, 9202.2109375
3: -2300.4028320, 5857.5903320, -1853.0117188, 4781.1162109, -7081.5190430, 7710.6020508
4: -6584.4262695, 4295.7910156, -5364.6601562, 3465.7893066, -10050.2158203, 9660.4472656

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1945753
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1892375
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6425.7265625, 5167.1157227, -2952.6821289, 2402.8015137, -8828.5283203, 8119.7968750
1: -5155.9707031, 4968.6342773, -2365.3845215, 2314.5366211, -7470.5073242, 7334.0185547
2: -7384.1015625, 5399.9472656, -3383.4587402, 2517.0407715, -9901.1425781, 8783.4042969
3: -2848.9968262, 7299.4448242, -1326.4201660, 3353.4309082, -6202.4277344, 8625.8652344
4: -8198.9033203, 5322.7695312, -3765.1281738, 2475.0305176, -10673.9335938, 9087.8964844

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4646323, upper bound: 4552.1884538
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1861417
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6465.6679688, 5197.9145508, -4267.7421875, 3427.3356934, -9893.0039062, 9465.6562500
1: -5188.6606445, 4998.1103516, -3433.8293457, 3298.9086914, -8487.5683594, 8431.9394531
2: -7431.6689453, 5431.6601562, -4925.3828125, 3575.4016113, -11007.0703125, 10357.0410156
3: -2865.7429199, 7345.2402344, -1881.0106201, 4863.0942383, -7724.6611328, 9226.2509766
4: -8250.3779297, 5354.0131836, -5456.5566406, 3519.1264648, -11769.5009766, 10810.5673828

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1898073
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1872713
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -5158.3984375, 4167.5395508, -9325.9375000, 9325.9375000
1: -4135.0488281, 4011.9240723, -4135.0488281, 4011.9240723, -8146.9726562, 8146.9726562
2: -5915.9174805, 4360.9184570, -5915.9174805, 4360.9184570, -10274.1767578, 10274.1767578
3: -2300.4028320, 5857.5903320, -2300.4028320, 5857.5903320, -8157.9863281, 8157.9863281
4: -6584.4262695, 4295.7910156, -6584.4262695, 4295.7910156, -10875.1572266, 10875.1562500

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1922033
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1868654
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -6423.7919922, 5165.6123047, -10320.3984375, 10591.3320312
1: -4135.0488281, 4011.9240723, -5154.3891602, 4967.1982422, -9100.9150391, 9166.3134766
2: -5915.9174805, 4360.9184570, -7381.7993164, 5398.4008789, -11305.0205078, 11734.5673828
3: -2300.4028320, 5857.5903320, -2848.1799316, 7297.2226562, -9577.5800781, 8702.2695312
4: -6584.4262695, 4295.7910156, -8196.4091797, 5321.2441406, -11893.0292969, 12487.0957031

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1945753
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1892375
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6423.7919922, 5165.6123047, -5158.3984375, 4167.5395508, -10591.3320312, 10320.3974609
1: -5154.3891602, 4967.1982422, -4135.0488281, 4011.9240723, -9166.3134766, 9100.9150391
2: -7381.7993164, 5398.4008789, -5915.9174805, 4360.9184570, -11734.5673828, 11305.0195312
3: -2848.1799316, 7297.2226562, -2300.4028320, 5857.5903320, -8702.2695312, 9577.5800781
4: -8196.4091797, 5321.2441406, -6584.4262695, 4295.7910156, -12487.0957031, 11893.0292969

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4646323, upper bound: 4552.1819100
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844829, upper bound: 4552.1795979
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6465.6679688, 5197.9145508, -6465.6679688, 5197.9145508, -11659.2441406, 11659.2441406
1: -5188.6606445, 4998.1103516, -5188.6606445, 4998.1103516, -10185.4218750, 10185.4208984
2: -7431.6689453, 5431.6601562, -7431.6689453, 5431.6601562, -12847.5341797, 12847.5341797
3: -2865.7429199, 7345.2402344, -2865.7429199, 7345.2402344, -10186.5107422, 10186.5107422
4: -8250.3779297, 5354.0131836, -8250.3779297, 5354.0131836, -13591.3330078, 13591.3330078

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1898002
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1872592
time: 0.87 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.94 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.94
Output dim: 0, lower bound: -4551.6986709, upper bound: 4551.7638377
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.94
Output dim: 0, lower bound: -4551.7920228, upper bound: 4551.7659535
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.94
Output dim: 0, lower bound: -4551.6986709, upper bound: 4551.7638377
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.94
Output dim: 0, lower bound: -4551.7920228, upper bound: 4551.7638377
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.94
Output dim: 0, lower bound: -4551.7282293, upper bound: 4551.7583069
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.94
Output dim: 0, lower bound: -4551.7536229, upper bound: 4551.7536229
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.94
Output dim: 0, lower bound: -4551.7282293, upper bound: 4551.7583069
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.94
Output dim: 0, lower bound: -4551.7536229, upper bound: 4551.7536229
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.94
Output dim: 0, lower bound: -4551.6986709, upper bound: 4551.7637713
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.94
Output dim: 0, lower bound: -4551.7985891, upper bound: 4551.7658871
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.94
Output dim: 0, lower bound: -4551.6986709, upper bound: 4551.7638109
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.94
Output dim: 0, lower bound: -4551.7985891, upper bound: 4551.7659267
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.94
Output dim: 0, lower bound: -4551.7347956, upper bound: 4551.7582405
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.94
Output dim: 0, lower bound: -4551.7601892, upper bound: 4551.7535565
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.94
Output dim: 0, lower bound: -4551.7347956, upper bound: 4551.7582713
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.94
Output dim: 0, lower bound: -4551.7601892, upper bound: 4551.7535565
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1924999
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1871620
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1945753
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1892375
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -4551.4646323, upper bound: 4552.1884538
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1861417
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1898073
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1872713
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1922033
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1868654
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1945753
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1892375
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -4551.4646323, upper bound: 4552.1819100
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -4552.1844829, upper bound: 4552.1795979
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1898002
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1872592

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -2952.6821289, 2402.8015137, -6999.9189453, 6660.2255859
1: -3687.0437012, 3569.1484375, -2365.3845215, 2314.5366211, -6001.5800781, 5934.5332031
2: -5276.2031250, 3879.7956543, -3383.4587402, 2517.0407715, -7793.2426758, 7263.2524414
3: -2047.8087158, 5221.5805664, -1326.4201660, 3353.4309082, -5401.2392578, 6547.9995117
4: -5870.9716797, 3822.0715332, -3765.1281738, 2475.0305176, -8346.0019531, 7587.1992188

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7380623, upper bound: 4551.7103340
time: 1.91 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7401350, upper bound: 4551.8036859
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -2952.6821289, 2402.8015137, -7506.7626953, 7075.9648438
1: -4091.2307129, 3969.2382812, -2365.3845215, 2314.5366211, -6405.7670898, 6334.6230469
2: -5853.0830078, 4314.3935547, -3383.4587402, 2517.0407715, -8370.1240234, 7697.8510742
3: -2275.4304199, 5795.1562500, -1326.4201660, 3353.4309082, -5628.8608398, 7121.5761719
4: -6514.6850586, 4250.0151367, -3765.1281738, 2475.0305176, -8989.7158203, 8015.1425781

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7679201, upper bound: 4551.7029061
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7699927, upper bound: 4551.7962579
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -4188.4218750, 3367.0412598, -7964.1586914, 7895.9653320
1: -3687.0437012, 3569.1484375, -3368.3234863, 3240.5815430, -6927.6240234, 6937.4716797
2: -5276.2031250, 3879.7956543, -4830.4160156, 3513.3398438, -8789.5429688, 8710.2119141
3: -2047.8087158, 5221.5805664, -1849.3717041, 4770.5156250, -6818.3237305, 7070.9516602
4: -5870.9716797, 3822.0715332, -5352.7685547, 3458.8666992, -9329.8378906, 9174.8398438

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7322538, upper bound: 4551.7398922
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7280324, upper bound: 4551.7652860
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -4188.0439453, 3366.7546387, -8470.7138672, 8311.3271484
1: -4091.2307129, 3969.2382812, -3368.0124512, 3240.3037109, -7331.5341797, 7337.2509766
2: -5853.0830078, 4314.3935547, -4829.9658203, 3513.0446777, -9366.1279297, 9144.3593750
3: -2275.4304199, 5795.1562500, -1849.2207031, 4770.0761719, -7045.5058594, 7644.3769531
4: -6514.6850586, 4250.0151367, -5352.2763672, 3458.5795898, -9973.2646484, 9602.2910156

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7621116, upper bound: 4551.7324642
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7578902, upper bound: 4551.7578580
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6325.6704102, 5088.0839844, -2952.6821289, 2402.8015137, -8728.4716797, 8040.7661133
1: -5077.3623047, 4893.3222656, -2365.3845215, 2314.5366211, -7391.8989258, 7258.7070312
2: -7274.0000000, 5316.0483398, -3383.4587402, 2517.0407715, -9791.0410156, 8699.5068359
3: -2804.0725098, 7193.1835938, -1326.4201660, 3353.4309082, -6157.5034180, 8519.6035156
4: -8074.3090820, 5239.3300781, -3765.1281738, 2475.0305176, -10549.3388672, 9004.4570312

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4292608, upper bound: 4551.6875449
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4343192, upper bound: 4551.7947080
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6350.0063477, 5106.2768555, -2952.6821289, 2402.8015137, -8752.8076172, 8058.9589844
1: -5095.0336914, 4909.9272461, -2365.3845215, 2314.5366211, -7409.5703125, 7275.3115234
2: -7296.8901367, 5335.7216797, -3383.4587402, 2517.0407715, -9813.9287109, 8719.1796875
3: -2814.9750977, 7214.0942383, -1326.4201660, 3353.4309082, -6168.4062500, 8540.5146484
4: -8102.4345703, 5259.8027344, -3765.1281738, 2475.0305176, -10577.4648438, 9024.9306641

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7427411, upper bound: 4551.6863623
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7477994, upper bound: 4551.7935255
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6366.1660156, 5119.2890625, -4267.7421875, 3427.3356934, -9793.5019531, 9387.0302734
1: -5110.7729492, 4923.2451172, -3433.8293457, 3298.9086914, -8409.6816406, 8357.0742188
2: -7322.4482422, 5348.0849609, -4925.3828125, 3575.4016113, -10897.8486328, 10273.4677734
3: -2820.9536133, 7239.8808594, -1881.0106201, 4863.0942383, -7676.4799805, 9120.8203125
4: -8126.7597656, 5270.9204102, -5456.5566406, 3519.1264648, -11645.8867188, 10727.4765625

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4241378, upper bound: 4551.7171103
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4219976, upper bound: 4551.7437677
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6390.3413086, 5137.2963867, -4267.7421875, 3427.3356934, -9817.6767578, 9405.0390625
1: -5128.1879883, 4939.5961914, -3433.8293457, 3298.9086914, -8427.0957031, 8373.4257812
2: -7344.9951172, 5367.6391602, -4925.3828125, 3575.4016113, -10920.3964844, 10293.0214844
3: -2831.8232422, 7260.3583984, -1881.0106201, 4863.0942383, -7687.5229492, 9141.3691406
4: -8154.4711914, 5291.2568359, -5456.5566406, 3519.1264648, -11673.5957031, 10747.8134766

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7376181, upper bound: 4551.7160190
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7354778, upper bound: 4551.7425851
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -5158.3984375, 4167.5395508, -8764.6572266, 8865.9423828
1: -3687.0437012, 3569.1484375, -4135.0488281, 4011.9240723, -7698.9667969, 7704.1972656
2: -5276.2031250, 3879.7956543, -5915.9174805, 4360.9184570, -9632.9892578, 9791.7519531
3: -2047.8087158, 5221.5805664, -2300.4028320, 5857.5903320, -7904.4311523, 7519.7099609
4: -5870.9716797, 3822.0715332, -6584.4262695, 4295.7910156, -10160.3740234, 10400.6279297

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1621650, upper bound: 4552.1621650
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1621650, upper bound: 4552.1916969
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -5158.3984375, 4167.5395508, -9271.5009766, 9281.6806641
1: -4091.2307129, 3969.2382812, -4135.0488281, 4011.9240723, -8103.1542969, 8104.2871094
2: -5853.0830078, 4314.3935547, -5915.9174805, 4360.9184570, -10211.1474609, 10227.9863281
3: -2275.4304199, 5795.1562500, -2300.4028320, 5857.5903320, -8133.0200195, 8094.7338867
4: -6514.6850586, 4250.0151367, -6584.4262695, 4295.7910156, -10805.3154297, 10829.7656250

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1916969, upper bound: 4552.1621650
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1916969, upper bound: 4552.1916969
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -6416.8925781, 5160.2539062, -9752.1845703, 10124.4355469
1: -3687.0437012, 3569.1484375, -5148.7817383, 4962.0717773, -8646.8906250, 8717.9296875
2: -5276.2031250, 3879.7956543, -7373.5825195, 5392.8813477, -10658.3242188, 11244.0791016
3: -2047.8087158, 5221.5805664, -2845.2658691, 7289.2915039, -9316.2392578, 8061.0878906
4: -5870.9716797, 3822.0715332, -8187.5156250, 5315.8071289, -11172.8164062, 12003.7246094

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1545275, upper bound: 4551.4742097
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1513498, upper bound: 4552.1940604
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -6416.6069336, 5160.0322266, -10260.3027344, 10539.8876953
1: -4091.2307129, 3969.2382812, -5148.5507812, 4961.8598633, -9051.6748047, 9117.7890625
2: -5853.0830078, 4314.3935547, -7373.2431641, 5392.6523438, -11236.2548828, 11679.9814453
3: -2275.4304199, 5795.1562500, -2845.1455078, 7288.9633789, -9544.6894531, 8635.9912109
4: -6514.6850586, 4250.0151367, -8187.1469727, 5315.5830078, -11817.5332031, 12432.4970703

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1794140, upper bound: 4551.4667226
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1762363, upper bound: 4552.1865733
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6323.7397461, 5086.5961914, -5158.3984375, 4167.5395508, -10491.1191406, 10238.5693359
1: -5075.7695312, 4891.8984375, -4135.0488281, 4011.9240723, -9087.6933594, 9022.8193359
2: -7271.6938477, 5314.5219727, -5915.9174805, 4360.9184570, -11621.3320312, 11217.4423828
3: -2803.2668457, 7190.9619141, -2300.4028320, 5857.5903320, -8653.9990234, 9470.7939453
4: -8071.8125000, 5237.8227539, -6584.4262695, 4295.7910156, -12359.8554688, 11806.1455078

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4742097, upper bound: 4552.1554065
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4667226, upper bound: 4552.1802930
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6348.0751953, 5104.7900391, -5158.3984375, 4167.5395508, -10515.6152344, 10256.4912109
1: -5093.4555664, 4908.5039062, -4135.0488281, 4011.9240723, -9105.3798828, 9039.4873047
2: -7294.5927734, 5334.1909180, -5915.9174805, 4360.9184570, -11648.2294922, 11237.1972656
3: -2814.1660156, 7211.8833008, -2300.4028320, 5857.5903320, -8665.0722656, 9492.1738281
4: -8099.9482422, 5258.2949219, -6584.4262695, 4295.7910156, -12391.7382812, 11826.3369141

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1940604, upper bound: 4552.1529031
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1865733, upper bound: 4552.1777897
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6366.1660156, 5119.2890625, -6465.6679688, 5197.9145508, -11557.8154297, 11577.7519531
1: -5110.7729492, 4923.2451172, -5188.6606445, 4998.1103516, -10105.2763672, 10107.7431641
2: -7322.4482422, 5348.0849609, -7431.6689453, 5431.6601562, -12735.1406250, 12760.1982422
3: -2820.9536133, 7239.8808594, -2865.7429199, 7345.2402344, -10138.3300781, 10080.5986328
4: -8126.7597656, 5270.9204102, -8250.3779297, 5354.0131836, -13465.0605469, 13504.7070312

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4646321, upper bound: 4551.4668377
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4646321, upper bound: 4552.1872591
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6390.3413086, 5137.2963867, -6465.6679688, 5197.9145508, -11584.8183594, 11595.4892578
1: -5128.1879883, 4939.5961914, -5188.6606445, 4998.1103516, -10125.4355469, 10124.1347656
2: -7344.9951172, 5367.6391602, -7431.6689453, 5431.6601562, -12761.6904297, 12779.8447266
3: -2831.8232422, 7260.3583984, -2865.7429199, 7345.2402344, -10149.3730469, 10101.5341797
4: -8154.4711914, 5291.2568359, -8250.3779297, 5354.0131836, -13496.5107422, 13524.7314453

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1762363, upper bound: 4551.4668377
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1872591
time: 0.90 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.21 seconds
IS_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.21
Output dim: 0, lower bound: -4551.7380623, upper bound: 4551.7103340
IS_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.21
Output dim: 0, lower bound: -4551.7401350, upper bound: 4551.8036859
IS_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.21
Output dim: 0, lower bound: -4551.7679201, upper bound: 4551.7029061
IS_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.21
Output dim: 0, lower bound: -4551.7699927, upper bound: 4551.7962579
IS_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.21
Output dim: 0, lower bound: -4551.7322538, upper bound: 4551.7398922
IS_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.21
Output dim: 0, lower bound: -4551.7280324, upper bound: 4551.7652860
IS_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.21
Output dim: 0, lower bound: -4551.7621116, upper bound: 4551.7324642
IS_A2_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.21
Output dim: 0, lower bound: -4551.7578902, upper bound: 4551.7578580
IS_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.21
Output dim: 0, lower bound: -4550.4292608, upper bound: 4551.6875449
IS_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.21
Output dim: 0, lower bound: -4550.4343192, upper bound: 4551.7947080
IS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.21
Output dim: 0, lower bound: -4551.7427411, upper bound: 4551.6863623
IS_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.21
Output dim: 0, lower bound: -4551.7477994, upper bound: 4551.7935255
IS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.21
Output dim: 0, lower bound: -4550.4241378, upper bound: 4551.7171103
IS_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.21
Output dim: 0, lower bound: -4550.4219976, upper bound: 4551.7437677
IS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.21
Output dim: 0, lower bound: -4551.7376181, upper bound: 4551.7160190
IS_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.21
Output dim: 0, lower bound: -4551.7354778, upper bound: 4551.7425851
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -4552.1621650, upper bound: 4552.1621650
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -4552.1621650, upper bound: 4552.1916969
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -4552.1916969, upper bound: 4552.1621650
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -4552.1916969, upper bound: 4552.1916969
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -4552.1545275, upper bound: 4551.4742097
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -4552.1513498, upper bound: 4552.1940604
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -4552.1794140, upper bound: 4551.4667226
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -4552.1762363, upper bound: 4552.1865733
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -4551.4742097, upper bound: 4552.1554065
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -4551.4667226, upper bound: 4552.1802930
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -4552.1940604, upper bound: 4552.1529031
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -4552.1865733, upper bound: 4552.1777897
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.21
Output dim: 0, lower bound: -4551.4646321, upper bound: 4551.4668377
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -4551.4646321, upper bound: 4552.1872591
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -4552.1762363, upper bound: 4551.4668377
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1872591

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -4597.1176758, 3707.5437012, -8304.6611328, 8304.6611328
1: -3687.0437012, 3569.1484375, -3687.0437012, 3569.1484375, -7256.1918945, 7256.1918945
2: -5276.2031250, 3879.7956543, -5276.2031250, 3879.7956543, -9150.5644531, 9150.5644531
3: -2047.8087158, 5221.5805664, -2047.8087158, 5221.5805664, -7266.1542969, 7266.1542969
4: -5870.9716797, 3822.0715332, -5870.9716797, 3822.0715332, -9685.8447266, 9685.8437500

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.7139066, upper bound: 4551.7342625
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.7139066, upper bound: 4551.7397307
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -5103.9614258, 4123.2832031, -8720.4003906, 8811.5039062
1: -3687.0437012, 3569.1484375, -4091.2307129, 3969.2382812, -7656.2822266, 7660.3789062
2: -5276.2031250, 3879.7956543, -5853.0830078, 4314.3935547, -9586.7988281, 9728.7226562
3: -2047.8087158, 5221.5805664, -2275.4304199, 5795.1562500, -7841.1791992, 7494.9267578
4: -5870.9716797, 3822.0715332, -6514.6850586, 4250.0151367, -10114.9833984, 10330.7871094

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.7139066, upper bound: 4551.7641203
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7323025, upper bound: 4551.7695884
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -4597.1176758, 3707.5437012, -8811.5029297, 8720.4003906
1: -4091.2307129, 3969.2382812, -3687.0437012, 3569.1484375, -7660.3789062, 7656.2822266
2: -5853.0830078, 4314.3935547, -5276.2031250, 3879.7956543, -9728.7226562, 9586.7988281
3: -2275.4304199, 5795.1562500, -2047.8087158, 5221.5805664, -7494.9267578, 7841.1787109
4: -6514.6850586, 4250.0151367, -5870.9716797, 3822.0715332, -10330.7861328, 10114.9833984

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6117698, upper bound: 4551.7330142
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7621604, upper bound: 4551.7323027
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -5103.9614258, 4123.2832031, -9227.2421875, 9227.2431641
1: -4091.2307129, 3969.2382812, -4091.2307129, 3969.2382812, -8060.4687500, 8060.4687500
2: -5853.0830078, 4314.3935547, -5853.0830078, 4314.3935547, -10164.9560547, 10164.9570312
3: -2275.4304199, 5795.1562500, -2275.4304199, 5795.1562500, -8069.9511719, 8069.9506836
4: -6514.6850586, 4250.0151367, -6514.6850586, 4250.0151367, -10759.9257812, 10759.9248047

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6117698, upper bound: 4551.7623046
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7621604, upper bound: 4551.7590118
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -6316.8413086, 5081.2817383, -9670.4101562, 10023.4853516
1: -3687.0437012, 3569.1484375, -5070.0815430, 4886.8125000, -8568.8320312, 8639.2304688
2: -5276.2031250, 3879.7956543, -7263.4555664, 5309.0625000, -10570.8193359, 11130.8193359
3: -2047.8087158, 5221.5805664, -2800.3874512, 7183.0278320, -9209.4492188, 8012.8583984
4: -5870.9716797, 3822.0715332, -8062.8930664, 5232.4389648, -11085.9980469, 11876.4609375

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4445115, upper bound: 4550.4219729
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4968185, upper bound: 4550.4290304
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -6341.1791992, 5099.4780273, -9688.3330078, 10048.7226562
1: -3687.0437012, 3569.1484375, -5087.8198242, 4903.4223633, -8585.5068359, 8656.9677734
2: -5276.2031250, 3879.7956543, -7286.3925781, 5328.7216797, -10590.5615234, 11157.7578125
3: -2047.8087158, 5221.5805664, -2811.2773438, 7203.9882812, -9230.8710938, 8023.9204102
4: -5870.9716797, 3822.0715332, -8091.0761719, 5252.9067383, -11106.1914062, 11908.3916016

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4444965, upper bound: 4551.7388089
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7091438, upper bound: 4551.7470782
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -6316.5537109, 5081.0620117, -10178.5292969, 10439.8359375
1: -4091.2307129, 3969.2382812, -5069.8461914, 4886.6025391, -8973.6171875, 9039.0839844
2: -5853.0830078, 4314.3935547, -7263.1127930, 5308.8359375, -11148.7519531, 11566.7187500
3: -2275.4304199, 5795.1562500, -2800.2678223, 7182.6987305, -9437.9003906, 8587.7626953
4: -6514.6850586, 4250.0151367, -8062.5224609, 5232.2163086, -11730.7167969, 12305.2324219

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.1823506, upper bound: 4550.4219544
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.4984063, upper bound: 4550.4219562
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -6340.8950195, 5099.2578125, -10196.4560547, 10464.1748047
1: -4091.2307129, 3969.2382812, -5087.5864258, 4903.2114258, -8990.2958984, 9056.8242188
2: -5853.0830078, 4314.3935547, -7286.0532227, 5328.4946289, -11168.4970703, 11593.6582031
3: -2275.4304199, 5795.1562500, -2811.1579590, 7203.6621094, -9459.3242188, 8598.8242188
4: -6514.6850586, 4250.0151367, -8090.7089844, 5252.6835938, -11750.9130859, 12337.1650391

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.5933311, upper bound: 4551.7388911
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7376065, upper bound: 4551.7391450
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6316.8413086, 5081.2817383, -4597.1176758, 3707.5437012, -10023.4853516, 9670.4101562
1: -5070.0815430, 4886.8125000, -3687.0437012, 3569.1484375, -8639.2304688, 8568.8320312
2: -7263.4555664, 5309.0625000, -5276.2031250, 3879.7956543, -11130.8193359, 10570.8183594
3: -2800.3874512, 7183.0278320, -2047.8087158, 5221.5805664, -8012.8583984, 9209.4501953
4: -8062.8930664, 5232.4389648, -5870.9716797, 3822.0715332, -11876.4609375, 11085.9980469

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.8832384, upper bound: 4551.7139337
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4302495, upper bound: 4551.7125020
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6316.5537109, 5081.0620117, -5103.9614258, 4123.2832031, -10439.8359375, 10178.5302734
1: -5069.8461914, 4886.6025391, -4091.2307129, 3969.2382812, -9039.0839844, 8973.6171875
2: -7263.1127930, 5308.8359375, -5853.0830078, 4314.3935547, -11566.7197266, 11148.7509766
3: -2800.2678223, 7182.6987305, -2275.4304199, 5795.1562500, -8587.7626953, 9437.9003906
4: -8062.5224609, 5232.2163086, -6514.6850586, 4250.0151367, -12305.2324219, 11730.7158203

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.8753052, upper bound: 4551.7423962
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4223163, upper bound: 4551.7409645
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6341.1791992, 5099.4780273, -4597.1176758, 3707.5437012, -10048.7226562, 9688.3330078
1: -5087.8198242, 4903.4223633, -3687.0437012, 3569.1484375, -8656.9677734, 8585.5078125
2: -7286.3925781, 5328.7216797, -5276.2031250, 3879.7956543, -11157.7578125, 10590.5625000
3: -2811.2773438, 7203.9882812, -2047.8087158, 5221.5805664, -8023.9204102, 9230.8710938
4: -8091.0761719, 5252.9067383, -5870.9716797, 3822.0715332, -11908.3906250, 11106.1923828

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7229013, upper bound: 4551.7124846
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7470782, upper bound: 4551.7124870
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6340.8950195, 5099.2578125, -5103.9614258, 4123.2832031, -10464.1748047, 10196.4560547
1: -5087.5864258, 4903.2114258, -4091.2307129, 3969.2382812, -9056.8242188, 8990.2958984
2: -7286.0532227, 5328.4946289, -5853.0830078, 4314.3935547, -11593.6582031, 11168.4970703
3: -2811.1579590, 7203.6621094, -2275.4304199, 5795.1562500, -8598.8251953, 9459.3242188
4: -8090.7089844, 5252.6835938, -6514.6850586, 4250.0151367, -12337.1660156, 11750.9130859

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7149680, upper bound: 4551.7409471
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7391450, upper bound: 4551.7409495
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6366.1660156, 5119.2890625, -6390.3413086, 5137.2963867, -11494.0605469, 11503.3271484
1: -5110.7729492, 4923.2451172, -5128.1879883, 4939.5961914, -10043.9892578, 10047.7578125
2: -7322.4482422, 5348.0849609, -7344.9951172, 5367.6391602, -12667.4521484, 12674.3535156
3: -2820.9536133, 7239.8808594, -2831.8232422, 7260.3583984, -10053.3544922, 10043.4609375
4: -8126.7597656, 5270.9204102, -8154.4711914, 5291.2568359, -13398.4580078, 13409.8847656

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.7561070, upper bound: 4551.7420811
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4258744, upper bound: 4551.7417620
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6390.3413086, 5137.2963867, -6366.1660156, 5119.2890625, -11503.3271484, 11494.0615234
1: -5128.1879883, 4939.5961914, -5110.7729492, 4923.2451172, -10047.7568359, 10043.9892578
2: -7344.9951172, 5367.6391602, -7322.4482422, 5348.0849609, -12674.3544922, 12667.4521484
3: -2831.8232422, 7260.3583984, -2820.9536133, 7239.8808594, -10043.4609375, 10053.3535156
4: -8154.4711914, 5291.2568359, -8126.7597656, 5270.9204102, -13409.8847656, 13398.4580078

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.5549222, upper bound: 4550.4258699
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.5417017, upper bound: 4550.4273814
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6390.3413086, 5137.2963867, -6390.3413086, 5137.2963867, -11521.0634766, 11521.0644531
1: -5128.1879883, 4939.5961914, -5128.1879883, 4939.5961914, -10064.1494141, 10064.1494141
2: -7344.9951172, 5367.6391602, -7344.9951172, 5367.6391602, -12694.0019531, 12694.0019531
3: -2831.8232422, 7260.3583984, -2831.8232422, 7260.3583984, -10064.3964844, 10064.3964844
4: -8154.4711914, 5291.2568359, -8154.4711914, 5291.2568359, -13429.9082031, 13429.9091797

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.5549223, upper bound: 4551.7333355
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.5417021, upper bound: 4551.7375208
time: 0.96 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.45 seconds
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -4550.7139066, upper bound: 4551.7342625
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -4550.7139066, upper bound: 4551.7397307
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -4550.7139066, upper bound: 4551.7641203
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -4551.7323025, upper bound: 4551.7695884
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -4551.6117698, upper bound: 4551.7330142
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -4551.7621604, upper bound: 4551.7323027
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -4551.6117698, upper bound: 4551.7623046
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -4551.7621604, upper bound: 4551.7590118
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -4550.4445115, upper bound: 4550.4219729
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -4551.4968185, upper bound: 4550.4290304
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -4550.4444965, upper bound: 4551.7388089
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -4551.7091438, upper bound: 4551.7470782
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -4551.1823506, upper bound: 4550.4219544
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -4551.4984063, upper bound: 4550.4219562
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -4551.5933311, upper bound: 4551.7388911
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -4551.7376065, upper bound: 4551.7391450
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -4550.8832384, upper bound: 4551.7139337
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -4550.4302495, upper bound: 4551.7125020
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -4550.8753052, upper bound: 4551.7423962
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -4550.4223163, upper bound: 4551.7409645
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -4551.7229013, upper bound: 4551.7124846
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -4551.7470782, upper bound: 4551.7124870
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -4551.7149680, upper bound: 4551.7409471
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -4551.7391450, upper bound: 4551.7409495
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -4550.7561070, upper bound: 4551.7420811
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -4550.4258744, upper bound: 4551.7417620
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -4551.5549222, upper bound: 4550.4258699
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -4551.5417017, upper bound: 4550.4273814
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -4551.5549223, upper bound: 4551.7333355
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -4551.5417021, upper bound: 4551.7375208
Binary search (step 3): status=Status.VERIFIED, low=0.9375000, high=1.0000000, mid=0.9375000, abs_max=5687.5751953125
rel_dist={0: [-4552.289001211693, 4552.289001211693]}

## Binary search (step 4) starts
Candidate diff: 0.9687500


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2865166, upper bound: 4552.2783404
time: 0.69 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711
time: 1.09 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.92 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.92
Output dim: 0, lower bound: -4552.2865166, upper bound: 4552.2783404
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.92
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -2987.9899902, 2430.9243164, -3136.6748047, 2550.9006348, -5538.8906250, 5567.5991211
1: -2393.8061523, 2341.6025391, -2513.2941895, 2457.4460449, -4851.2519531, 4854.8964844
2: -3424.2626953, 2546.4648438, -3595.4755859, 2672.1398926, -6096.4023438, 6141.9404297
3: -1341.7878418, 3393.8977051, -1407.5072021, 3564.2558594, -4906.0439453, 4801.4047852
4: -3810.2302246, 2503.8750000, -3999.4067383, 2627.0041504, -6437.2343750, 6503.2817383

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711
time: 0.85 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711
time: 0.93 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -5188.6577148, 4191.8315430, -3136.6748047, 2550.9006348, -7739.5585938, 7328.5063477
1: -4159.4165039, 4035.3022461, -2513.2941895, 2457.4460449, -6616.8623047, 6548.5966797
2: -5950.7929688, 4386.3300781, -3595.4755859, 2672.1398926, -8622.9326172, 7981.8056641
3: -2313.8149414, 5892.1474609, -1407.5072021, 3564.2558594, -5878.0708008, 7299.6542969
4: -6623.0805664, 4320.7236328, -3999.4067383, 2627.0041504, -9250.0830078, 8320.1308594

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711
time: 1.53 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711
time: 0.84 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.00 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.00
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.00
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.00
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.00
Output dim: 0, lower bound: -4552.2764711, upper bound: 4552.2764711

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -2987.9899902, 2430.9243164, -2987.9899902, 2430.9243164, -5418.9140625, 5418.9140625
1: -2393.8061523, 2341.6025391, -2393.8061523, 2341.6025391, -4735.4086914, 4735.4086914
2: -3424.2626953, 2546.4648438, -3424.2626953, 2546.4648438, -5970.7275391, 5970.7275391
3: -1341.7878418, 3393.8977051, -1341.7878418, 3393.8977051, -4735.6855469, 4735.6855469
4: -3810.2302246, 2503.8750000, -3810.2302246, 2503.8750000, -6314.1054688, 6314.1054688

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2085840, upper bound: 4552.1951258
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1880421
time: 0.77 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -2987.9899902, 2430.9243164, -5188.6577148, 4191.8315430, -7179.8212891, 7619.5820312
1: -2393.8061523, 2341.6025391, -4159.4165039, 4035.3022461, -6429.1083984, 6501.0190430
2: -3424.2626953, 2546.4648438, -5950.7929688, 4386.3300781, -7810.5922852, 8497.2558594
3: -1341.7878418, 3393.8977051, -2313.8149414, 5892.1474609, -7233.9355469, 5707.7128906
4: -3810.2302246, 2503.8750000, -6623.0805664, 4320.7236328, -8130.9526367, 9126.9550781

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.2085840, upper bound: 4552.1951258
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1880421
time: 0.89 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -5188.6577148, 4191.8315430, -2987.9899902, 2430.9243164, -7619.5820312, 7179.8212891
1: -4159.4165039, 4035.3022461, -2393.8061523, 2341.6025391, -6501.0190430, 6429.1083984
2: -5950.7929688, 4386.3300781, -3424.2626953, 2546.4648438, -8497.2558594, 7810.5922852
3: -2313.8149414, 5892.1474609, -1341.7878418, 3393.8977051, -5707.7128906, 7233.9355469
4: -6623.0805664, 4320.7236328, -3810.2302246, 2503.8750000, -9126.9550781, 8130.9531250

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1946649
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1905380
time: 0.88 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -5188.6577148, 4191.8315430, -5188.6577148, 4191.8315430, -9380.4892578, 9380.4892578
1: -4159.4165039, 4035.3022461, -4159.4165039, 4035.3022461, -8194.7187500, 8194.7187500
2: -5950.7929688, 4386.3300781, -5950.7929688, 4386.3300781, -10334.5771484, 10334.5771484
3: -2313.8149414, 5892.1474609, -2313.8149414, 5892.1474609, -8205.9628906, 8205.9628906
4: -6623.0805664, 4320.7236328, -6623.0805664, 4320.7236328, -10938.7910156, 10938.7919922

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1946649
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1905380
time: 0.84 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.59 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.59
Output dim: 0, lower bound: -4552.2085840, upper bound: 4552.1951258
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.59
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1880421
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.59
Output dim: 0, lower bound: -4552.2085840, upper bound: 4552.1951258
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.59
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1880421
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.59
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1946649
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.59
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1905380
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.59
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1946649
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.59
Output dim: 0, lower bound: -4552.1880420, upper bound: 4552.1905380

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -2987.9899902, 2430.9243164, -5383.6064453, 5390.7915039
1: -2365.3845215, 2314.5366211, -2393.8061523, 2341.6025391, -4706.9873047, 4708.3427734
2: -3383.4587402, 2517.0407715, -3424.2626953, 2546.4648438, -5929.9228516, 5941.3027344
3: -1326.4201660, 3353.4309082, -1341.7878418, 3393.8977051, -4720.3173828, 4695.2187500
4: -3765.1281738, 2475.0305176, -3810.2302246, 2503.8750000, -6269.0029297, 6285.2602539

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1859666, upper bound: 4552.1859666
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1859666, upper bound: 4552.1880421
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4215.9218750, 3387.9650879, -2987.9899902, 2430.9243164, -6646.8461914, 6375.9545898
1: -3391.0078125, 3260.8413086, -2393.8061523, 2341.6025391, -5732.6103516, 5654.6474609
2: -4863.3217773, 3534.8996582, -3424.2626953, 2546.4648438, -7409.7866211, 6959.1616211
3: -1860.3760986, 4802.6064453, -1341.7878418, 3393.8977051, -5254.2739258, 6144.3945312
4: -5388.7382812, 3479.7985840, -3810.2302246, 2503.8750000, -7892.6132812, 7290.0283203

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1859666
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1880421
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -5188.6577148, 4191.8315430, -7144.5136719, 7591.4589844
1: -2365.3845215, 2314.5366211, -4159.4165039, 4035.3022461, -6400.6865234, 6473.9531250
2: -3383.4587402, 2517.0407715, -5950.7929688, 4386.3300781, -7769.7875977, 8467.8320312
3: -1326.4201660, 3353.4309082, -2313.8149414, 5892.1474609, -7218.5668945, 5667.2460938
4: -3765.1281738, 2475.0305176, -6623.0805664, 4320.7236328, -8085.8510742, 9098.1093750

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1884626, upper bound: 4552.1856700
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1884626, upper bound: 4552.1880421
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4213.7329102, 3386.2946777, -5188.6577148, 4191.8315430, -8405.5644531, 8574.9521484
1: -3389.2006836, 3259.2282715, -4159.4165039, 4035.3022461, -7424.5029297, 7418.6445312
2: -4860.7001953, 3533.1818848, -5950.7929688, 4386.3300781, -9247.0302734, 9483.9746094
3: -1859.5002441, 4800.0473633, -2313.8149414, 5892.1474609, -7751.6474609, 7113.8623047
4: -5385.8730469, 3478.1318359, -6623.0805664, 4320.7236328, -9706.5957031, 10101.2119141

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905380, upper bound: 4552.1856700
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905380, upper bound: 4552.1880421
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -2987.9899902, 2430.9243164, -7589.3227539, 7155.5288086
1: -4135.0488281, 4011.9240723, -2393.8061523, 2341.6025391, -6476.6513672, 6405.7290039
2: -5915.9174805, 4360.9184570, -3424.2626953, 2546.4648438, -8462.3818359, 7785.1806641
3: -2300.4028320, 5857.5903320, -1341.7878418, 3393.8977051, -5694.3007812, 7199.3779297
4: -6584.4262695, 4295.7910156, -3810.2302246, 2503.8750000, -9088.3007812, 8106.0195312

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1884626
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1905380
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6437.8808594, 5176.5029297, -2987.9899902, 2430.9243164, -8868.8046875, 8164.4921875
1: -5165.9008789, 4977.6108398, -2393.8061523, 2341.6025391, -7507.5034180, 7371.4165039
2: -7398.5610352, 5409.6142578, -3424.2626953, 2546.4648438, -9945.0244141, 8833.8769531
3: -2854.1035156, 7313.3720703, -1341.7878418, 3393.8977051, -6248.0009766, 8655.1601562
4: -8214.5527344, 5332.2915039, -3810.2302246, 2503.8750000, -10718.4277344, 9142.5214844

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1884626
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1905380
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -5188.6577148, 4191.8315430, -9350.2294922, 9356.1972656
1: -4135.0488281, 4011.9240723, -4159.4165039, 4035.3022461, -8170.3510742, 8171.3408203
2: -5915.9174805, 4360.9184570, -5950.7929688, 4386.3300781, -10299.6093750, 10309.1445312
3: -2300.4028320, 5857.5903320, -2313.8149414, 5892.1474609, -8192.5507812, 8171.4052734
4: -6584.4262695, 4295.7910156, -6623.0805664, 4320.7236328, -10900.1123047, 10913.8359375

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1881660
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1905380
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6436.1958008, 5175.2050781, -5188.6577148, 4191.8315430, -10628.0263672, 10360.2548828
1: -5164.5249023, 4976.3686523, -4159.4165039, 4035.3022461, -9199.8242188, 9134.4609375
2: -7396.5566406, 5408.2768555, -5950.7929688, 4386.3300781, -11774.4853516, 11349.8437500
3: -2853.3969727, 7311.4423828, -2313.8149414, 5892.1474609, -8742.1962891, 9604.9853516
4: -8212.3847656, 5330.9746094, -6623.0805664, 4320.7236328, -12527.9287109, 11941.4277344

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1881660
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1905380
time: 0.76 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.15 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -4552.1859666, upper bound: 4552.1859666
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -4552.1859666, upper bound: 4552.1880421
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1859666
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1880421
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -4552.1884626, upper bound: 4552.1856700
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -4552.1884626, upper bound: 4552.1880421
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -4552.1905380, upper bound: 4552.1856700
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -4552.1905380, upper bound: 4552.1880421
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1884626
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1905380
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1884626
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1905380
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1881660
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -4552.1856700, upper bound: 4552.1905380
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1881660
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -4552.1880421, upper bound: 4552.1905380

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -2952.6821289, 2402.8015137, -5355.4833984, 5355.4833984
1: -2365.3845215, 2314.5366211, -2365.3845215, 2314.5366211, -4679.9208984, 4679.9208984
2: -3383.4587402, 2517.0407715, -3383.4587402, 2517.0407715, -5900.4965820, 5900.4970703
3: -1326.4201660, 3353.4309082, -1326.4201660, 3353.4309082, -4679.8505859, 4679.8505859
4: -3765.1281738, 2475.0305176, -3765.1281738, 2475.0305176, -6240.1582031, 6240.1582031

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6986709, upper bound: 4551.7638377
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7920228, upper bound: 4551.7659535
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -4200.5449219, 3376.2512207, -6328.9331055, 6603.3466797
1: -2365.3845215, 2314.5366211, -3378.3203125, 3249.5102539, -5614.8945312, 5692.8569336
2: -3383.4587402, 2517.0407715, -4844.9140625, 3522.8425293, -6906.3002930, 7361.9536133
3: -1326.4201660, 3353.4309082, -1854.2230225, 4784.6445312, -6111.0639648, 5207.6538086
4: -3765.1281738, 2475.0305176, -5368.6186523, 3468.0932617, -7233.2197266, 7843.6494141

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.6986709, upper bound: 4551.7638377
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7920228, upper bound: 4551.7659535
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4200.5449219, 3376.2512207, -2952.6821289, 2402.8015137, -6603.3466797, 6328.9331055
1: -3378.3203125, 3249.5102539, -2365.3845215, 2314.5366211, -5692.8569336, 5614.8945312
2: -4844.9140625, 3522.8425293, -3383.4587402, 2517.0407715, -7361.9536133, 6906.3002930
3: -1854.2230225, 4784.6445312, -1326.4201660, 3353.4309082, -5207.6538086, 6111.0644531
4: -5368.6186523, 3468.0932617, -3765.1281738, 2475.0305176, -7843.6494141, 7233.2197266

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7282291, upper bound: 4551.7583069
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7536229, upper bound: 4551.7536229
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4267.7421875, 3427.3356934, -4267.7421875, 3427.3356934, -7695.0771484, 7695.0771484
1: -3433.8293457, 3298.9086914, -3433.8293457, 3298.9086914, -6732.7377930, 6732.7377930
2: -4925.3828125, 3575.4016113, -4925.3828125, 3575.4016113, -8500.7832031, 8500.7841797
3: -1881.0106201, 4863.0942383, -1881.0106201, 4863.0942383, -6744.1049805, 6744.1049805
4: -5456.5566406, 3519.1264648, -5456.5566406, 3519.1264648, -8975.6835938, 8975.6835938

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7282293, upper bound: 4551.7583069
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7536229, upper bound: 4551.7536229
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -5158.3984375, 4167.5395508, -7120.2216797, 7561.2001953
1: -2365.3845215, 2314.5366211, -4135.0488281, 4011.9240723, -6377.3085938, 6449.5854492
2: -3383.4587402, 2517.0407715, -5915.9174805, 4360.9184570, -7744.3754883, 8432.9580078
3: -1326.4201660, 3353.4309082, -2300.4028320, 5857.5903320, -7184.0102539, 5653.8339844
4: -3765.1281738, 2475.0305176, -6584.4262695, 4295.7910156, -8060.9179688, 9059.4570312

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7052373, upper bound: 4551.7637713
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7985891, upper bound: 4551.7658871
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -2952.6821289, 2402.8015137, -6426.0947266, 5167.4003906, -8120.0825195, 8828.8964844
1: -2365.3845215, 2314.5366211, -5156.2705078, 4968.9067383, -7334.2910156, 7470.8071289
2: -3383.4587402, 2517.0407715, -7384.5390625, 5400.2392578, -8783.6982422, 9901.5781250
3: -1326.4201660, 3353.4309082, -2849.1513672, 7299.8662109, -8626.2851562, 6202.5820312
4: -3765.1281738, 2475.0305176, -8199.3759766, 5323.0571289, -9088.1845703, 10674.4042969

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7052373, upper bound: 4551.7638109
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7985891, upper bound: 4551.7659267
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4198.0107422, 3374.3251953, -5158.3984375, 4167.5395508, -8365.5507812, 8532.7236328
1: -3376.2299805, 3247.6435547, -4135.0488281, 4011.9240723, -7388.1542969, 7382.6923828
2: -4841.8833008, 3520.8559570, -5915.9174805, 4360.9184570, -9202.8007812, 9436.7734375
3: -1853.2092285, 4781.6899414, -2300.4028320, 5857.5903320, -7710.7988281, 7082.0927734
4: -5365.3051758, 3466.1647949, -6584.4262695, 4295.7910156, -9661.0937500, 10050.5908203

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7347956, upper bound: 4551.7582405
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7601892, upper bound: 4551.7535565
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4267.7421875, 3427.3356934, -6465.6679688, 5197.9145508, -9465.6562500, 9893.0039062
1: -3433.8293457, 3298.9086914, -5188.6606445, 4998.1103516, -8431.9394531, 8487.5683594
2: -4925.3828125, 3575.4016113, -7431.6689453, 5431.6601562, -10357.0410156, 11007.0703125
3: -1881.0106201, 4863.0942383, -2865.7429199, 7345.2402344, -9226.2509766, 7724.6606445
4: -5456.5566406, 3519.1264648, -8250.3779297, 5354.0131836, -10810.5673828, 11769.5009766

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7347956, upper bound: 4551.7582713
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7601892, upper bound: 4551.7535565
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -2952.6821289, 2402.8015137, -7561.2001953, 7120.2216797
1: -4135.0488281, 4011.9240723, -2365.3845215, 2314.5366211, -6449.5854492, 6377.3085938
2: -5915.9174805, 4360.9184570, -3383.4587402, 2517.0407715, -8432.9580078, 7744.3750000
3: -2300.4028320, 5857.5903320, -1326.4201660, 3353.4309082, -5653.8339844, 7184.0102539
4: -6584.4262695, 4295.7910156, -3765.1281738, 2475.0305176, -9059.4570312, 8060.9179688

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1924999
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1871620
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -4198.0107422, 3374.3251953, -8532.7236328, 8365.5507812
1: -4135.0488281, 4011.9240723, -3376.2299805, 3247.6435547, -7382.6923828, 7388.1542969
2: -5915.9174805, 4360.9184570, -4841.8833008, 3520.8559570, -9436.7734375, 9202.8017578
3: -2300.4028320, 5857.5903320, -1853.2092285, 4781.6899414, -7082.0927734, 7710.7988281
4: -6584.4262695, 4295.7910156, -5365.3051758, 3466.1647949, -10050.5908203, 9661.0937500

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1945753
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1892375
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6426.0947266, 5167.4003906, -2952.6821289, 2402.8015137, -8828.8964844, 8120.0825195
1: -5156.2705078, 4968.9067383, -2365.3845215, 2314.5366211, -7470.8071289, 7334.2910156
2: -7384.5390625, 5400.2392578, -3383.4587402, 2517.0407715, -9901.5781250, 8783.6982422
3: -2849.1513672, 7299.8662109, -1326.4201660, 3353.4309082, -6202.5820312, 8626.2861328
4: -8199.3759766, 5323.0571289, -3765.1281738, 2475.0305176, -10674.4042969, 9088.1845703

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4646321, upper bound: 4552.1884538
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1762363, upper bound: 4552.1861417
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6465.6679688, 5197.9145508, -4267.7421875, 3427.3356934, -9893.0039062, 9465.6562500
1: -5188.6606445, 4998.1103516, -3433.8293457, 3298.9086914, -8487.5683594, 8431.9394531
2: -7431.6689453, 5431.6601562, -4925.3828125, 3575.4016113, -11007.0703125, 10357.0410156
3: -2865.7429199, 7345.2402344, -1881.0106201, 4863.0942383, -7724.6611328, 9226.2509766
4: -8250.3779297, 5354.0131836, -5456.5566406, 3519.1264648, -11769.5009766, 10810.5673828

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1898073
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1872713
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -5158.3984375, 4167.5395508, -9325.9375000, 9325.9375000
1: -4135.0488281, 4011.9240723, -4135.0488281, 4011.9240723, -8146.9726562, 8146.9726562
2: -5915.9174805, 4360.9184570, -5915.9174805, 4360.9184570, -10274.1767578, 10274.1767578
3: -2300.4028320, 5857.5903320, -2300.4028320, 5857.5903320, -8157.9863281, 8157.9863281
4: -6584.4262695, 4295.7910156, -6584.4262695, 4295.7910156, -10875.1572266, 10875.1562500

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1922033
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1868654
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5158.3984375, 4167.5395508, -6424.1669922, 5165.9028320, -10320.6884766, 10591.7070312
1: -4135.0488281, 4011.9240723, -5154.6958008, 4967.4755859, -9101.1933594, 9166.6201172
2: -5915.9174805, 4360.9184570, -7382.2456055, 5398.6997070, -11305.3193359, 11735.0068359
3: -2300.4028320, 5857.5903320, -2848.3378906, 7297.6523438, -9578.0039062, 8702.4257812
4: -6584.4262695, 4295.7910156, -8196.8916016, 5321.5400391, -11893.3232422, 12487.5751953

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1945753
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1892375
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6424.1669922, 5165.9028320, -5158.3984375, 4167.5395508, -10591.7070312, 10320.6875000
1: -5154.6958008, 4967.4755859, -4135.0488281, 4011.9240723, -9166.6201172, 9101.1943359
2: -7382.2456055, 5398.6997070, -5915.9174805, 4360.9184570, -11735.0068359, 11305.3183594
3: -2848.3378906, 7297.6523438, -2300.4028320, 5857.5903320, -8702.4267578, 9578.0039062
4: -8196.8916016, 5321.5400391, -6584.4262695, 4295.7910156, -12487.5742188, 11893.3242188

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4551.4646321, upper bound: 4552.1819100
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1795979
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6465.6679688, 5197.9145508, -6465.6679688, 5197.9145508, -11659.2441406, 11659.2441406
1: -5188.6606445, 4998.1103516, -5188.6606445, 4998.1103516, -10185.4218750, 10185.4208984
2: -7431.6689453, 5431.6601562, -7431.6689453, 5431.6601562, -12847.5341797, 12847.5341797
3: -2865.7429199, 7345.2402344, -2865.7429199, 7345.2402344, -10186.5107422, 10186.5107422
4: -8250.3779297, 5354.0131836, -8250.3779297, 5354.0131836, -13591.3330078, 13591.3330078

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1
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
time: 0.72 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.02 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.02
Output dim: 0, lower bound: -4551.6986709, upper bound: 4551.7638377
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.02
Output dim: 0, lower bound: -4551.7920228, upper bound: 4551.7659535
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.02
Output dim: 0, lower bound: -4551.6986709, upper bound: 4551.7638377
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.02
Output dim: 0, lower bound: -4551.7920228, upper bound: 4551.7659535
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.02
Output dim: 0, lower bound: -4551.7282291, upper bound: 4551.7583069
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.02
Output dim: 0, lower bound: -4551.7536229, upper bound: 4551.7536229
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.02
Output dim: 0, lower bound: -4551.7282293, upper bound: 4551.7583069
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.02
Output dim: 0, lower bound: -4551.7536229, upper bound: 4551.7536229
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.02
Output dim: 0, lower bound: -4551.7052373, upper bound: 4551.7637713
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.02
Output dim: 0, lower bound: -4551.7985891, upper bound: 4551.7658871
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.02
Output dim: 0, lower bound: -4551.7052373, upper bound: 4551.7638109
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.02
Output dim: 0, lower bound: -4551.7985891, upper bound: 4551.7659267
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.02
Output dim: 0, lower bound: -4551.7347956, upper bound: 4551.7582405
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.02
Output dim: 0, lower bound: -4551.7601892, upper bound: 4551.7535565
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.02
Output dim: 0, lower bound: -4551.7347956, upper bound: 4551.7582713
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.02
Output dim: 0, lower bound: -4551.7601892, upper bound: 4551.7535565
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.02
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1924999
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.02
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1871620
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.02
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1945753
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.02
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1892375
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.02
Output dim: 0, lower bound: -4551.4646321, upper bound: 4552.1884538
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.02
Output dim: 0, lower bound: -4552.1762363, upper bound: 4552.1861417
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.02
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1898073
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.02
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1872713
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.02
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1922033
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.02
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1868654
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.02
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1945753
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.02
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1892375
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.02
Output dim: 0, lower bound: -4551.4646321, upper bound: 4552.1819100
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.02
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1795979
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.02
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1898002
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.02
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1872592

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -2952.6821289, 2402.8015137, -6999.9189453, 6660.2255859
1: -3687.0437012, 3569.1484375, -2365.3845215, 2314.5366211, -6001.5800781, 5934.5332031
2: -5276.2031250, 3879.7956543, -3383.4587402, 2517.0407715, -7793.2426758, 7263.2524414
3: -2047.8087158, 5221.5805664, -1326.4201660, 3353.4309082, -5401.2392578, 6547.9995117
4: -5870.9716797, 3822.0715332, -3765.1281738, 2475.0305176, -8346.0019531, 7587.1992188

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7380623, upper bound: 4551.7103340
time: 2.09 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7401350, upper bound: 4551.8036859
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -2952.6821289, 2402.8015137, -7506.7626953, 7075.9648438
1: -4091.2307129, 3969.2382812, -2365.3845215, 2314.5366211, -6405.7670898, 6334.6230469
2: -5853.0830078, 4314.3935547, -3383.4587402, 2517.0407715, -8370.1240234, 7697.8510742
3: -2275.4304199, 5795.1562500, -1326.4201660, 3353.4309082, -5628.8608398, 7121.5761719
4: -6514.6850586, 4250.0151367, -3765.1281738, 2475.0305176, -8989.7158203, 8015.1425781

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7679201, upper bound: 4551.7029061
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7699927, upper bound: 4551.7962579
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4597.1176758, 3707.5437012, -4188.9565430, 3367.4465332, -7964.5644531, 7896.5000000
1: -3687.0437012, 3569.1484375, -3368.7644043, 3240.9753418, -6928.0180664, 6937.9130859
2: -5276.2031250, 3879.7956543, -4831.0561523, 3513.7583008, -8789.9609375, 8710.8515625
3: -2047.8087158, 5221.5805664, -1849.5854492, 4771.1381836, -6818.9467773, 7071.1660156
4: -5870.9716797, 3822.0715332, -5353.4672852, 3459.2734375, -9330.2431641, 9175.5390625

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7322538, upper bound: 4551.7398922
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7280324, upper bound: 4551.7652860
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5103.9614258, 4123.2832031, -4188.5820312, 3367.1621094, -8471.1210938, 8311.8652344
1: -4091.2307129, 3969.2382812, -3368.4550781, 3240.6989746, -7331.9287109, 7337.6933594
2: -5853.0830078, 4314.3935547, -4830.6069336, 3513.4643555, -9366.5468750, 9145.0000000
3: -2275.4304199, 5795.1562500, -1849.4351807, 4770.7011719, -7046.1303711, 7644.5913086
4: -6514.6850586, 4250.0151367, -5352.9775391, 3458.9880371, -9973.6728516, 9602.9912109

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7621116, upper bound: 4551.7324642
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7578902, upper bound: 4551.7578580
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6326.0356445, 5088.3652344, -2952.6821289, 2402.8015137, -8728.8369141, 8041.0468750
1: -5077.6635742, 4893.5917969, -2365.3845215, 2314.5366211, -7392.2001953, 7258.9765625
2: -7274.4365234, 5316.3374023, -3383.4587402, 2517.0407715, -9791.4775391, 8699.7949219
3: -2804.2250977, 7193.6030273, -1326.4201660, 3353.4309082, -6157.6562500, 8520.0234375
4: -8074.7822266, 5239.6147461, -3765.1281738, 2475.0305176, -10549.8115234, 9004.7421875

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4292608, upper bound: 4551.6875449
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4550.4343192, upper bound: 4551.7947080
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6350.3725586, 5106.5585938, -2952.6821289, 2402.8015137, -8753.1738281, 8059.2402344
1: -5095.3310547, 4910.1953125, -2365.3845215, 2314.5366211, -7409.8676758, 7275.5800781
2: -7297.3256836, 5336.0122070, -3383.4587402, 2517.0407715, -9814.3652344, 8719.4697266
3: -2815.1284180, 7214.5131836, -1326.4201660, 3353.4309082, -6168.5590820, 8540.9335938
4: -8102.9038086, 5260.0878906, -3765.1281738, 2475.0305176, -10577.9326172, 9025.2158203

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7427411, upper bound: 4551.6863623
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4551.7477994, upper bound: 4551.7935255
time: 0.82 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.23 seconds
IS_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 0, lower bound: -4551.7380623, upper bound: 4551.7103340
IS_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 0, lower bound: -4551.7401350, upper bound: 4551.8036859
IS_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 0, lower bound: -4551.7679201, upper bound: 4551.7029061
IS_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 0, lower bound: -4551.7699927, upper bound: 4551.7962579
IS_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 0, lower bound: -4551.7322538, upper bound: 4551.7398922
IS_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 0, lower bound: -4551.7280324, upper bound: 4551.7652860
IS_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 0, lower bound: -4551.7621116, upper bound: 4551.7324642
IS_A2_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 0, lower bound: -4551.7578902, upper bound: 4551.7578580
IS_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 0, lower bound: -4550.4292608, upper bound: 4551.6875449
IS_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 0, lower bound: -4550.4343192, upper bound: 4551.7947080
IS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 0, lower bound: -4551.7427411, upper bound: 4551.6863623
IS_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 0, lower bound: -4551.7477994, upper bound: 4551.7935255
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1898073
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1872713
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1922033
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1868654
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -4552.1609696, upper bound: 4552.1945753
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -4552.1905015, upper bound: 4552.1892375
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -4551.4646321, upper bound: 4552.1819100
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1795979
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -4551.4646324, upper bound: 4552.1898002
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -4552.1844830, upper bound: 4552.1872592
Binary search (step 4): status=Status.UNKNOWN, low=0.9375000, high=0.9687500, mid=0.9687500, abs_max=5687.5751953125
rel_dist={0: [-4552.289001211694, 4552.289001211693]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.9375
execution time: 1121.03 seconds
