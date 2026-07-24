## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2000 seconds
Threshold: 10.310653145
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540)
1: (-4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358)
2: (-5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447)
3: (-6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520)
4: (-6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002)
5: (-5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936)
6: (-4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792)
7: (-5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823)
8: (-7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416)
9: (-4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449)

## BASE Result
execution time: IAR + LP analysis = 1.42 + 5.30 = 6.72 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -10.8533233, upper bound: 10.8533231


# Binary Search by BASE starts (time budget: 1993.28 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=12.009641647338867
rel_dist={8: [-10.853321756160986, 10.853321760348969]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=12.009641647338867
rel_dist={8: [-10.85331894689149, 10.853318913028158]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=12.009641647338867
rel_dist={8: [-10.853314892972527, 10.853314942404314]}

## Binary Search Result
Binary search time: 21.05 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1972.23 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8478131, upper bound: 10.8508153
time: 2.54 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8476960, upper bound: 10.8476960
time: 2.29 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 4.98 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 4.98
Output dim: 8, lower bound: -10.8478131, upper bound: 10.8508153
IS_A2, status: Status.UNKNOWN, split count: 1, time: 4.98
Output dim: 8, lower bound: -10.8476960, upper bound: 10.8476960

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -4.8091030, 3.9809673, -5.1619492, 4.2683043, -9.0774078, 9.1429167
1: -4.1900167, 3.7616804, -4.5150156, 4.0219212, -8.2119379, 8.2766953
2: -5.4816332, 3.8969140, -5.8974509, 4.1410961, -9.6227283, 9.7943640
3: -5.9486446, 3.4754472, -6.4188118, 3.7199407, -9.6685848, 9.8942585
4: -5.6237993, 4.1171808, -6.0448523, 4.4138484, -10.0376472, 10.1620331
5: -4.7429023, 3.9461298, -5.1020646, 4.2303286, -8.9732304, 9.0481949
6: -4.5476189, 4.4447222, -4.8802671, 4.7729130, -9.3205318, 9.3249893
7: -4.8612185, 4.7321424, -5.2257671, 5.0835156, -9.9447346, 9.9579096
8: -7.4275122, 3.8547773, -7.9439735, 4.0656700, -11.4931812, 11.7987490
9: -4.2459145, 4.5239472, -4.5706940, 4.8500509, -9.0959654, 9.0946398

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2996984, upper bound: 10.3367045
time: 2.98 seconds

## Relational analysis of IS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7710604, upper bound: 10.4899072
time: 2.04 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8440929, upper bound: 10.8454266
time: 3.97 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -4.8939848, 4.0257545, -4.8157530, 3.9869628, -8.8809471, 8.8415070
1: -4.2509437, 3.7665713, -4.1968932, 3.7674007, -8.0183439, 7.9634647
2: -5.5441694, 3.9399633, -5.4899254, 3.9023726, -9.4465389, 9.4298887
3: -5.8212814, 3.2330990, -5.9598937, 3.4847975, -9.3060789, 9.1929913
4: -5.6733103, 4.1140733, -5.6321068, 4.1230135, -9.7963238, 9.7461796
5: -4.7948580, 3.9627540, -4.7503901, 3.9524677, -8.7473249, 8.7131443
6: -4.5430226, 4.4767122, -4.5545239, 4.4513569, -8.9943790, 9.0312328
7: -4.9111648, 4.7876940, -4.8679743, 4.7388420, -9.6500072, 9.6556683
8: -7.5964932, 3.8544226, -7.4388371, 3.8568153, -11.4533081, 11.2932596
9: -4.2639904, 4.5530343, -4.2520156, 4.5311999, -8.7951908, 8.8050499

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6196327, upper bound: 10.4518837
time: 1.72 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8406201, upper bound: 10.8406201
time: 1.63 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 10.85 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 10.85
Output dim: 8, lower bound: -10.7710604, upper bound: 10.4899072
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 10.85
Output dim: 8, lower bound: -10.8440929, upper bound: 10.8454266
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 10.85
Output dim: 8, lower bound: -10.6196327, upper bound: 10.4518837
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 10.85
Output dim: 8, lower bound: -10.8406201, upper bound: 10.8406201

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -4.1009026, 3.4014392, -1.0109360, 1.0661721, -5.1670747, 4.4123755
1: -3.5279636, 3.2383189, -0.8210928, 0.9583632, -4.4863267, 4.0594120
2: -4.6369157, 3.3981578, -0.9250798, 1.2859070, -5.9228230, 4.3232374
3: -5.0063457, 2.9930849, -0.9066604, 1.0202264, -6.0265722, 3.8997455
4: -4.7703257, 3.5267489, -1.0571092, 0.9829249, -5.7532492, 4.5838580
5: -4.0147853, 3.3779554, -0.9172652, 1.0346648, -5.0494499, 4.2952209
6: -3.8734596, 3.7920160, -0.9817039, 0.9888433, -4.8623028, 4.7737198
7: -4.1339922, 4.0260501, -0.9717048, 1.0027300, -5.1367221, 4.9977551
8: -6.3777742, 3.4732904, -1.3888791, 2.6229784, -9.0007524, 4.8621693
9: -3.5967920, 3.8705828, -0.9480655, 1.1377145, -4.7345066, 4.8186483

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7327192, upper bound: 10.4762809
time: 2.72 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7327192, upper bound: 10.4899072
time: 2.94 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -4.8091030, 3.9809673, -3.7270885, 3.0890789, -7.8981810, 7.7080555
1: -4.1900167, 3.7616804, -3.1653371, 2.9619009, -7.1519165, 6.9270172
2: -5.4816332, 3.8969140, -4.1831040, 3.1329398, -8.6145725, 8.0800180
3: -5.9486446, 3.4754472, -4.5280991, 2.7820382, -8.7306824, 8.0035458
4: -5.6237993, 4.1171808, -4.3120480, 3.2161894, -8.8399868, 8.4292288
5: -4.7429023, 3.9461298, -3.6234443, 3.0854201, -7.8283224, 7.5695734
6: -4.5476189, 4.4447222, -3.5185735, 3.4486907, -7.9963078, 7.9632959
7: -4.8612185, 4.7321424, -3.7487261, 3.6495347, -8.5107536, 8.4808683
8: -7.4275122, 3.8547773, -5.8063211, 3.2942400, -10.7217522, 9.6610966
9: -4.2459145, 4.5239472, -3.2495794, 3.5275011, -7.7734156, 7.7735267

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8293682, upper bound: 10.8125995
time: 2.34 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8293682, upper bound: 10.8454267
time: 2.82 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -4.3015876, 3.5521805, -0.8025995, 0.9390517, -5.2406392, 4.3547802
1: -3.6964424, 3.3337524, -0.6627920, 0.8060393, -4.5024815, 3.9965444
2: -4.8397112, 3.5190885, -0.7264460, 1.1456316, -5.9853430, 4.2455344
3: -5.0568285, 2.8681016, -0.6567504, 0.8927312, -5.9495597, 3.5248520
4: -4.9629822, 3.6312871, -0.8367934, 0.8162170, -5.7791982, 4.4680805
5: -4.1866937, 3.4884472, -0.7391869, 0.8831133, -5.0698071, 4.2276340
6: -3.9891531, 3.9380550, -0.7824451, 0.8178614, -4.8070130, 4.7205000
7: -4.2986264, 4.1969233, -0.7901295, 0.8176407, -5.1162663, 4.9870529
8: -6.7069159, 3.5729733, -1.0297760, 2.5985899, -9.3055058, 4.6027479
9: -3.7295299, 4.0103531, -0.8191464, 0.9742404, -4.7037702, 4.8294992

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4166512, upper bound: 10.4166512
time: 1.36 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4166512, upper bound: 10.4518837
time: 1.48 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -4.8939848, 4.0257545, -3.4356151, 2.8430080, -7.7369909, 7.4613695
1: -4.2509437, 3.7665713, -2.8854160, 2.7439387, -6.9948816, 6.6519871
2: -5.5441694, 3.9399633, -3.8261924, 2.9269135, -8.4710827, 7.7661557
3: -5.8212814, 3.2330990, -4.1272268, 2.5835505, -8.4048319, 7.3603249
4: -5.6733103, 4.1140733, -3.9470525, 2.9688058, -8.6421146, 8.0611258
5: -4.7948580, 3.9627540, -3.3170941, 2.8510010, -7.6458588, 7.2798481
6: -4.5430226, 4.4767122, -3.2338035, 3.1755896, -7.7186122, 7.7105160
7: -4.9111648, 4.7876940, -3.4439979, 3.3514132, -8.2625780, 8.2316914
8: -7.5964932, 3.8544226, -5.3593159, 3.1775475, -10.7740402, 9.2137375
9: -4.2639904, 4.5530343, -2.9757266, 3.2557607, -7.5197511, 7.5287609

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4518837, upper bound: 10.6196327
time: 1.93 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4518837, upper bound: 10.8406201
time: 2.01 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 13.95 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 13.95
Output dim: 8, lower bound: -10.7327192, upper bound: 10.4762809
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 13.95
Output dim: 8, lower bound: -10.7327192, upper bound: 10.4899072
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 13.95
Output dim: 8, lower bound: -10.8293682, upper bound: 10.8125995
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 13.95
Output dim: 8, lower bound: -10.8293682, upper bound: 10.8454267
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 13.95
Output dim: 8, lower bound: -10.4166512, upper bound: 10.4166512
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 13.95
Output dim: 8, lower bound: -10.4166512, upper bound: 10.4518837
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 13.95
Output dim: 8, lower bound: -10.4518837, upper bound: 10.6196327
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 13.95
Output dim: 8, lower bound: -10.4518837, upper bound: 10.8406201

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.8224976, 0.9526591, -1.0109360, 1.0661721, -1.8886697, 1.9635952
1: -0.6783867, 0.8205066, -0.8210928, 0.9583632, -1.6367500, 1.6415994
2: -0.7437838, 1.1596640, -0.9250798, 1.2859070, -2.0296907, 2.0847437
3: -0.6810015, 0.9041843, -0.9066604, 1.0202264, -1.7012279, 1.8108448
4: -0.8567464, 0.8332369, -1.0571092, 0.9829249, -1.8396714, 1.8903461
5: -0.7553797, 0.8981278, -0.9172652, 1.0346648, -1.7900444, 1.8153930
6: -0.8005887, 0.8345573, -0.9817039, 0.9888433, -1.7894320, 1.8162613
7: -0.8075277, 0.8357326, -0.9717048, 1.0027300, -1.8102577, 1.8074374
8: -1.0656095, 2.6009374, -1.3888791, 2.6229784, -3.6885879, 3.9898164
9: -0.8306347, 0.9904779, -0.9480655, 1.1377145, -1.9683492, 1.9385433

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3551535, upper bound: 10.2674041
time: 1.97 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5181185, upper bound: 10.3600473
time: 2.66 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4587163, upper bound: 10.2182583
time: 2.18 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -3.4052205, 2.8175251, -1.0109360, 1.0661721, -4.4713926, 3.8284612
1: -2.8558207, 2.7208309, -0.8210928, 0.9583632, -3.8141837, 3.5419238
2: -3.7890821, 2.9046569, -0.9250798, 1.2859070, -5.0749893, 3.8297367
3: -4.0843210, 2.5608428, -0.9066604, 1.0202264, -5.1045475, 3.4675031
4: -3.9089458, 2.9427168, -1.0571092, 0.9829249, -4.8918705, 3.9998260
5: -3.2855368, 2.8261533, -0.9172652, 1.0346648, -4.3202014, 3.7434185
6: -3.2041881, 3.1468105, -0.9817039, 0.9888433, -4.1930313, 4.1285143
7: -3.4125867, 3.3201020, -0.9717048, 1.0027300, -4.4153166, 4.2918067
8: -5.3120508, 3.1667218, -1.3888791, 2.6229784, -7.9350290, 4.5556011
9: -2.9472203, 3.2273698, -0.9480655, 1.1377145, -4.0849347, 4.1754351

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3551535, upper bound: 10.2774174
time: 2.36 seconds

## Relational analysis of IS_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6010798, upper bound: 10.3428243
time: 2.36 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6010723, upper bound: 10.3427501
time: 2.57 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.8224976, 0.9526591, -3.7270885, 3.0890789, -3.9115765, 4.6797476
1: -0.6783867, 0.8205066, -3.1653371, 2.9619009, -3.6402876, 3.9858437
2: -0.7437838, 1.1596640, -4.1831040, 3.1329398, -3.8767238, 5.3427682
3: -0.6810015, 0.9041843, -4.5280991, 2.7820382, -3.4630396, 5.4322834
4: -0.8567464, 0.8332369, -4.3120480, 3.2161894, -4.0729361, 5.1452847
5: -0.7553797, 0.8981278, -3.6234443, 3.0854201, -3.8407998, 4.5215721
6: -0.8005887, 0.8345573, -3.5185735, 3.4486907, -4.2492795, 4.3531308
7: -0.8075277, 0.8357326, -3.7487261, 3.6495347, -4.4570622, 4.5844588
8: -1.0656095, 2.6009374, -5.8063211, 3.2942400, -4.3598495, 8.4072590
9: -0.8306347, 0.9904779, -3.2495794, 3.5275011, -4.3581357, 4.2400575

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 214

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3551535, upper bound: 10.6071825
time: 2.31 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5181185, upper bound: 10.6621857
time: 2.40 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4587163, upper bound: 10.5702073
time: 2.21 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -3.4052205, 2.8175251, -3.7270885, 3.0890789, -6.4942994, 6.5446138
1: -2.8558207, 2.7208309, -3.1653371, 2.9619009, -5.8177214, 5.8861680
2: -3.7890821, 2.9046569, -4.1831040, 3.1329398, -6.9220219, 7.0877609
3: -4.0843210, 2.5608428, -4.5280991, 2.7820382, -6.8663592, 7.0889416
4: -3.9089458, 2.9427168, -4.3120480, 3.2161894, -7.1251354, 7.2547646
5: -3.2855368, 2.8261533, -3.6234443, 3.0854201, -6.3709569, 6.4495974
6: -3.2041881, 3.1468105, -3.5185735, 3.4486907, -6.6528788, 6.6653843
7: -3.4125867, 3.3201020, -3.7487261, 3.6495347, -7.0621214, 7.0688281
8: -5.3120508, 3.1667218, -5.8063211, 3.2942400, -8.6062908, 8.9730434
9: -2.9472203, 3.2273698, -3.2495794, 3.5275011, -6.4747214, 6.4769492

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3551535, upper bound: 10.7167318
time: 2.06 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4786644, upper bound: 10.7751099
time: 2.28 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6982982, upper bound: 10.8317158
time: 3.45 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6010723, upper bound: 10.8106979
time: 3.70 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1.2885556, 1.2915748, -0.8025995, 0.9390517, -2.2276073, 2.0941744
1: -0.9978373, 1.1210864, -0.6627920, 0.8060393, -1.8038766, 1.7838783
2: -1.1817958, 1.4661613, -0.7264460, 1.1456316, -2.3274274, 2.1926074
3: -1.2152059, 1.0104641, -0.6567504, 0.8927312, -2.1079371, 1.6672144
4: -1.3690012, 1.1674459, -0.8367934, 0.8162170, -2.1852181, 2.0042393
5: -1.1490538, 1.2066720, -0.7391869, 0.8831133, -2.0321670, 1.9458590
6: -1.1961780, 1.2265388, -0.7824451, 0.8178614, -2.0140395, 2.0089839
7: -1.2147151, 1.2315087, -0.7901295, 0.8176407, -2.0323558, 2.0216384
8: -1.8933592, 2.8310375, -1.0297760, 2.5985899, -4.4919491, 3.8608136
9: -1.1436851, 1.3561463, -0.8191464, 0.9742404, -2.1179256, 2.1752927

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2618113, upper bound: 10.3132467
time: 2.21 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2498985, upper bound: 10.2498985
time: 1.54 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -3.6654465, 3.0519638, -0.8025995, 0.9390517, -4.6044984, 3.8545632
1: -3.0935688, 2.8719342, -0.6627920, 0.8060393, -3.8996081, 3.5347261
2: -4.0813146, 3.0818248, -0.7264460, 1.1456316, -5.2269459, 3.8082709
3: -4.2469306, 2.4954326, -0.6567504, 0.8927312, -5.1396618, 3.1521831
4: -4.2021399, 3.1118362, -0.8367934, 0.8162170, -5.0183568, 3.9486296
5: -3.5362399, 2.9846327, -0.7391869, 0.8831133, -4.4193530, 3.7238197
6: -3.4004350, 3.3648591, -0.7824451, 0.8178614, -4.2182965, 4.1473041
7: -3.6567678, 3.5623693, -0.7901295, 0.8176407, -4.4744086, 4.3524990
8: -5.7447891, 3.3173430, -1.0297760, 2.5985899, -8.3433790, 4.3471189
9: -3.1585369, 3.4368827, -0.8191464, 0.9742404, -4.1327772, 4.2560291

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1667724, upper bound: 10.3032319
time: 1.97 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1500091, upper bound: 10.1901466
time: 1.77 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1.2885556, 1.2915748, -3.4356151, 2.8430080, -4.1315637, 4.7271900
1: -0.9978373, 1.1210864, -2.8854160, 2.7439387, -3.7417758, 4.0065022
2: -1.1817958, 1.4661613, -3.8261924, 2.9269135, -4.1087093, 5.2923536
3: -1.2152059, 1.0104641, -4.1272268, 2.5835505, -3.7987564, 5.1376905
4: -1.3690012, 1.1674459, -3.9470525, 2.9688058, -4.3378067, 5.1144981
5: -1.1490538, 1.2066720, -3.3170941, 2.8510010, -4.0000548, 4.5237660
6: -1.1961780, 1.2265388, -3.2338035, 3.1755896, -4.3717675, 4.4603424
7: -1.2147151, 1.2315087, -3.4439979, 3.3514132, -4.5661283, 4.6755066
8: -1.8933592, 2.8310375, -5.3593159, 3.1775475, -5.0709066, 8.1903534
9: -1.1436851, 1.3561463, -2.9757266, 3.2557607, -4.3994455, 4.3318729

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2618113, upper bound: 10.4797376
time: 1.90 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2498985, upper bound: 10.4708484
time: 2.09 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -3.6654465, 3.0519638, -3.4356151, 2.8430080, -6.5084534, 6.4875789
1: -3.0935688, 2.8719342, -2.8854160, 2.7439387, -5.8375072, 5.7573500
2: -4.0813146, 3.0818248, -3.8261924, 2.9269135, -7.0082283, 6.9080172
3: -4.2469306, 2.4954326, -4.1272268, 2.5835505, -6.8304811, 6.6226597
4: -4.2021399, 3.1118362, -3.9470525, 2.9688058, -7.1709456, 7.0588884
5: -3.5362399, 2.9846327, -3.3170941, 2.8510010, -6.3872409, 6.3017268
6: -3.4004350, 3.3648591, -3.2338035, 3.1755896, -6.5760245, 6.5986624
7: -3.6567678, 3.5623693, -3.4439979, 3.3514132, -7.0081811, 7.0063672
8: -5.7447891, 3.3173430, -5.3593159, 3.1775475, -8.9223366, 8.6766586
9: -3.1585369, 3.4368827, -2.9757266, 3.2557607, -6.4142976, 6.4126091

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2618113, upper bound: 10.8146813
time: 2.03 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2498985, upper bound: 10.8002451
time: 2.00 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 23.69 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.69
Output dim: 8, lower bound: -10.5181185, upper bound: 10.3600473
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.69
Output dim: 8, lower bound: -10.4587163, upper bound: 10.2182583
IS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 23.69
Output dim: 8, lower bound: -10.6010798, upper bound: 10.3428243
IS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 23.69
Output dim: 8, lower bound: -10.6010723, upper bound: 10.3427501
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.69
Output dim: 8, lower bound: -10.5181185, upper bound: 10.6621857
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.69
Output dim: 8, lower bound: -10.4587163, upper bound: 10.5702073
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.69
Output dim: 8, lower bound: -10.6982982, upper bound: 10.8317158
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.69
Output dim: 8, lower bound: -10.6010723, upper bound: 10.8106979
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.69
Output dim: 8, lower bound: -10.2618113, upper bound: 10.3132467
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 23.69
Output dim: 8, lower bound: -10.2498985, upper bound: 10.2498985
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 23.69
Output dim: 8, lower bound: -10.1667724, upper bound: 10.3032319
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 23.69
Output dim: 8, lower bound: -10.1500091, upper bound: 10.1901466
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.69
Output dim: 8, lower bound: -10.2618113, upper bound: 10.4797376
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.69
Output dim: 8, lower bound: -10.2498985, upper bound: 10.4708484
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.69
Output dim: 8, lower bound: -10.2618113, upper bound: 10.8146813
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.69
Output dim: 8, lower bound: -10.2498985, upper bound: 10.8002451

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.8224976, 0.9526591, -0.6233644, 0.8078595, -1.6303571, 1.5760236
1: -0.6783867, 0.8205066, -0.5144333, 0.6521166, -1.3305033, 1.3349398
2: -0.7437838, 1.1596640, -0.5921754, 0.9798281, -1.7236120, 1.7518394
3: -0.6810015, 0.9041843, -0.4460992, 0.7376650, -1.4186665, 1.3502835
4: -0.8567464, 0.8332369, -0.6562716, 0.6501445, -1.5068910, 1.4895084
5: -0.7553797, 0.8981278, -0.5896373, 0.7228131, -1.4781928, 1.4877651
6: -0.8005887, 0.8345573, -0.6070594, 0.6720486, -1.4726373, 1.4416168
7: -0.8075277, 0.8357326, -0.6237040, 0.6413877, -1.4489154, 1.4594367
8: -1.0656095, 2.6009374, -0.6744797, 2.5529795, -3.6185889, 3.2754171
9: -0.8306347, 0.9904779, -0.7099549, 0.8203790, -1.6510136, 1.7004328

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3818682, upper bound: 10.1782777
time: 1.94 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3772685, upper bound: 10.1773845
time: 1.74 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.6678195, 0.8425452, -1.1396828, 1.1826501, -1.8504696, 1.9822279
1: -0.5538198, 0.6940145, -0.8952597, 1.0745120, -1.6283318, 1.5892742
2: -0.6215090, 1.0256290, -1.0344384, 1.3685452, -1.9900541, 2.0600674
3: -0.4967918, 0.7833476, -1.0428447, 1.0096173, -1.5064092, 1.8261923
4: -0.7006276, 0.6941398, -1.1928593, 1.0717310, -1.7723587, 1.8869991
5: -0.6267691, 0.7657658, -1.0201621, 1.1163161, -1.7430851, 1.7859279
6: -0.6526073, 0.7069265, -1.0783823, 1.1120255, -1.7646328, 1.7853087
7: -0.6669478, 0.6859019, -1.0716039, 1.1103120, -1.7772598, 1.7575058
8: -0.7662120, 2.5704176, -1.6408100, 2.6683919, -3.4346039, 4.2112274
9: -0.7350333, 0.8615387, -1.0403166, 1.2359829, -1.9710162, 1.9018552

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4078712, upper bound: 10.2036882
time: 2.95 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1754369, upper bound: 9.9582443
time: 3.35 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2369680, upper bound: 9.9744946
time: 2.20 seconds

## BFS IS instance: IS_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -2.6259260, 2.1722479, -1.0010154, 1.0601513, -3.6860774, 3.1732633
1: -2.1264014, 2.1283784, -0.8136675, 0.9507011, -3.0771027, 2.9420459
2: -2.8327496, 2.3362224, -0.9152174, 1.2790227, -4.1117725, 3.2514398
3: -3.0550425, 2.0664408, -0.8951954, 1.0144818, -4.0695243, 2.9616363
4: -2.9829547, 2.2829671, -1.0467370, 0.9751524, -3.9581070, 3.3297040
5: -2.4822688, 2.2194552, -0.9083072, 1.0276904, -3.5099592, 3.1277623
6: -2.4624128, 2.4206142, -0.9724166, 0.9804947, -3.4429076, 3.3930309
7: -2.6215091, 2.5364044, -0.9632244, 0.9943994, -3.6159084, 3.4996288
8: -4.0913000, 2.8535833, -1.3714924, 2.6206574, -6.7119575, 4.2250757
9: -2.2360466, 2.5191741, -0.9415510, 1.1298926, -3.3659391, 3.4607251

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 76

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4137022, upper bound: 10.1951768
time: 2.44 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6103264, upper bound: 10.3217569
time: 2.94 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6926487, upper bound: 10.3427501
time: 2.34 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6926487, upper bound: 10.3427501
time: 2.54 seconds

## BFS IS instance: IS_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -3.2783580, 2.7020874, -0.9425532, 1.0239800, -4.3023381, 3.6446407
1: -2.7150831, 2.6043346, -0.7701151, 0.9060723, -3.6211555, 3.3744497
2: -3.6297889, 2.7831078, -0.8576782, 1.2386779, -4.8684669, 3.6407859
3: -3.9183664, 2.4491372, -0.8265682, 0.9808536, -4.8992200, 3.2757053
4: -3.7575605, 2.8276746, -0.9857286, 0.9289462, -4.6865067, 3.8134031
5: -3.1550050, 2.7151055, -0.8562981, 0.9861840, -4.1411891, 3.5714035
6: -3.0729437, 3.0183504, -0.9170153, 0.9318503, -4.0047941, 3.9353657
7: -3.2861693, 3.1971188, -0.9128262, 0.9448866, -4.2310557, 4.1099448
8: -5.1023445, 3.0891998, -1.2690187, 2.6076651, -7.7100096, 4.3582182
9: -2.8239250, 3.1038258, -0.9029010, 1.0841417, -3.9080667, 4.0067267

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6928164, upper bound: 10.3427501
time: 2.08 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6928164, upper bound: 10.3427501
time: 2.38 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.8224976, 0.9526591, -2.9824953, 2.4663305, -3.2888281, 3.9351544
1: -0.6783867, 0.8205066, -2.4451320, 2.4074383, -3.0858250, 3.2656386
2: -0.7437838, 1.1596640, -3.2701421, 2.6048865, -3.3486705, 4.4298062
3: -0.6810015, 0.9041843, -3.5105333, 2.2920437, -2.9730451, 4.4147177
4: -0.8567464, 0.8332369, -3.3804235, 2.5840175, -3.4407640, 4.2136602
5: -0.7553797, 0.8981278, -2.8464975, 2.4899566, -3.2453363, 3.7446253
6: -0.8005887, 0.8345573, -2.7946393, 2.7521801, -3.5527687, 3.6291966
7: -0.8075277, 0.8357326, -2.9725847, 2.8864202, -3.6939478, 3.8083172
8: -1.0656095, 2.6009374, -4.6545305, 3.0066097, -4.0722189, 7.2554679
9: -0.8306347, 0.9904779, -2.5546801, 2.8379128, -3.6685474, 3.5451579

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1566564, upper bound: 10.2341987
time: 2.41 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5876467, upper bound: 10.5797660
time: 2.06 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5764508, upper bound: 10.5786747
time: 2.37 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.6678195, 0.8425452, -3.3906696, 2.8242786, -3.4920981, 4.2332149
1: -0.5538198, 0.6940145, -2.8663049, 2.7201111, -3.2739310, 3.5603194
2: -0.6215090, 1.0256290, -3.7490010, 2.8945532, -3.5160623, 4.7746301
3: -0.4967918, 0.7833476, -3.9561574, 2.4271555, -2.9239473, 4.7395048
4: -0.7006276, 0.6941398, -3.8802075, 2.9004204, -3.6010480, 4.5743475
5: -0.6267691, 0.7657658, -3.2646959, 2.7891145, -3.4158835, 4.0304618
6: -0.6526073, 0.7069265, -3.1540616, 3.1303453, -3.7829528, 3.8609881
7: -0.6669478, 0.6859019, -3.3787677, 3.2996078, -3.9665556, 4.0646696
8: -0.7662120, 2.5704176, -5.3142843, 3.1365914, -3.9028034, 7.8847017
9: -0.7350333, 0.8615387, -2.9314344, 3.1966333, -3.9316666, 3.7929730

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0957868, upper bound: 10.1513127
time: 2.44 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5689245, upper bound: 10.5164595
time: 1.99 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5569385, upper bound: 10.5111547
time: 2.13 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -3.3919077, 2.8062272, -2.9328561, 2.4127960, -5.8047037, 5.7390833
1: -2.8425677, 2.7105389, -2.3821862, 2.3555315, -5.1980991, 5.0927248
2: -3.7728319, 2.8946509, -3.2099998, 2.5388985, -6.3117304, 6.1046505
3: -4.0669127, 2.5523651, -3.4837213, 2.2754459, -6.3423586, 6.0360861
4: -3.8926854, 2.9314613, -3.3391724, 2.5442429, -6.4369283, 6.2706337
5: -3.2718637, 2.8155174, -2.8007469, 2.4521141, -5.7239780, 5.6162643
6: -3.1913819, 3.1342580, -2.7534752, 2.7012711, -5.8926530, 5.8877335
7: -3.3989973, 3.3067496, -2.9355774, 2.8502209, -6.2492180, 6.2423267
8: -5.2912846, 3.1605659, -4.5668526, 2.9443765, -8.2356606, 7.7274184
9: -2.9347944, 3.2151711, -2.5102952, 2.7934465, -5.7282410, 5.7254663

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8046807, upper bound: 10.6403004
time: 2.96 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8038779, upper bound: 10.6380652
time: 2.62 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -3.3107619, 2.7373619, -3.6088955, 2.9799416, -6.2907038, 6.3462572
1: -2.7623019, 2.6485758, -3.0334468, 2.8518283, -5.6141300, 5.6820226
2: -3.6737056, 2.8344827, -4.0335865, 3.0179672, -6.6916728, 6.8680692
3: -3.9604588, 2.5012145, -4.3737674, 2.6712751, -6.6317339, 6.8749819
4: -3.7930551, 2.8629100, -4.1703949, 3.1077013, -6.9007564, 7.0333052
5: -3.1882339, 2.7509828, -3.5007262, 2.9792786, -6.1675124, 6.2517090
6: -3.1132517, 3.0579391, -3.3946471, 3.3272810, -6.4405327, 6.4525862
7: -3.3158076, 3.2250323, -3.6294036, 3.5355167, -6.8513241, 6.8544359
8: -5.1644363, 3.1252160, -5.6099930, 3.2119558, -8.3763924, 8.7352085
9: -2.8594522, 3.1406913, -3.1330554, 3.4108317, -6.2702837, 6.2737465

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7307423, upper bound: 10.6686569
time: 2.50 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7936691, upper bound: 10.5910759
time: 1.96 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7927378, upper bound: 10.5883960
time: 3.42 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -1.2794669, 1.2859993, -0.4905117, 0.6848899, -1.9643568, 1.7765111
1: -0.9907948, 1.1143146, -0.4064204, 0.5249684, -1.5157633, 1.5207350
2: -1.1727405, 1.4598638, -0.5290366, 0.8079099, -1.9806504, 1.9889004
3: -1.2046871, 1.0058441, -0.3167189, 0.6403959, -1.8450830, 1.3225631
4: -1.3584956, 1.1606170, -0.5378752, 0.5320989, -1.8905945, 1.6984922
5: -1.1409898, 1.2001991, -0.4759714, 0.6139715, -1.7549613, 1.6761706
6: -1.1880946, 1.2189343, -0.4750422, 0.5601513, -1.7482460, 1.6939765
7: -1.2063935, 1.2241869, -0.5129223, 0.5216335, -1.7280270, 1.7371092
8: -1.8777492, 2.8287833, -0.4076656, 2.4894352, -4.3671846, 3.2364488
9: -1.1373957, 1.3489738, -0.6509184, 0.6906133, -1.8280090, 1.9998921

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 92

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -9.9972138, upper bound: 10.0115640
time: 1.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2

### Relational analysis result of IS_A2_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0184340, upper bound: 10.0793914
time: 1.35 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1.2794669, 1.2859993, -2.6585093, 2.1970522, -3.4765191, 3.9445086
1: -0.9907948, 1.1143146, -2.1530802, 2.1532009, -3.1439958, 3.2673948
2: -1.1727405, 1.4598638, -2.8731802, 2.3579764, -3.5307169, 4.3330441
3: -1.2046871, 1.0058441, -3.1016428, 2.0915184, -3.2962055, 4.1074867
4: -1.3584956, 1.1606170, -3.0197289, 2.3114562, -3.6699519, 4.1803460
5: -1.1409898, 1.2001991, -2.5162141, 2.2444091, -3.3853989, 3.7164133
6: -1.1880946, 1.2189343, -2.4935980, 2.4508598, -3.6389544, 3.7125323
7: -1.2063935, 1.2241869, -2.6546745, 2.5700183, -3.7764118, 3.8788614
8: -1.8777492, 2.8287833, -4.1429958, 2.8616734, -4.7394228, 6.9717789
9: -1.1373957, 1.3489738, -2.2643242, 2.5487087, -3.6861043, 3.6132979

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2419062, upper bound: 10.3515136
time: 1.83 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1088404, upper bound: 10.1840040
time: 2.08 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2944353, upper bound: 10.4708484
time: 2.31 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2944353, upper bound: 10.4708484
time: 1.55 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1.2250876, 1.2524472, -3.3111403, 2.7296522, -3.9547398, 4.5635877
1: -0.9484966, 1.0747626, -2.7470067, 2.6293330, -3.5778294, 3.8217692
2: -1.1185266, 1.4225944, -3.6699209, 2.8069715, -3.9254980, 5.0925155
3: -1.1414278, 0.9785590, -3.9647286, 2.4729557, -3.6143835, 4.9432878
4: -1.2953026, 1.1198655, -3.7989330, 2.8558354, -4.1511383, 4.9187984
5: -1.0925803, 1.1617638, -3.1892717, 2.7417488, -3.8343291, 4.3510356
6: -1.1401429, 1.1735252, -3.1050749, 3.0493367, -4.1894798, 4.2786002
7: -1.1564126, 1.1800897, -3.3201978, 3.2310126, -4.3874254, 4.5002875
8: -1.7844443, 2.8161662, -5.1533117, 3.0993311, -4.8837757, 7.9694777
9: -1.0997548, 1.3063253, -2.8547494, 3.1344552, -4.2342100, 4.1610737

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=12.009641647338867
rel_dist={8: [-10.853321756160986, 10.853321760348969]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8494828, upper bound: 10.8477722
time: 30.64 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8476955, upper bound: 10.8476954
time: 4.02 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 34.82 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 34.82
Output dim: 8, lower bound: -10.8494828, upper bound: 10.8477722
IS_B2, status: Status.UNKNOWN, split count: 1, time: 34.82
Output dim: 8, lower bound: -10.8476955, upper bound: 10.8476954

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -5.1619492, 4.2683043, -4.8091030, 3.9809673, -9.1429167, 9.0774078
1: -4.5150156, 4.0219212, -4.1900167, 3.7616804, -8.2766953, 8.2119370
2: -5.8974509, 4.1410961, -5.4816332, 3.8969140, -9.7943640, 9.6227293
3: -6.4188118, 3.7199407, -5.9486446, 3.4754472, -9.8942585, 9.6685848
4: -6.0448523, 4.4138484, -5.6237993, 4.1171808, -10.1620331, 10.0376472
5: -5.1020646, 4.2303286, -4.7429023, 3.9461298, -9.0481949, 8.9732304
6: -4.8802671, 4.7729130, -4.5476189, 4.4447222, -9.3249874, 9.3205318
7: -5.2257671, 5.0835156, -4.8612185, 4.7321424, -9.9579096, 9.9447346
8: -7.9439735, 4.0656700, -7.4275122, 3.8547773, -11.7987480, 11.4931812
9: -4.5706940, 4.8500509, -4.2459145, 4.5239472, -9.0946398, 9.0959644

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4588523, upper bound: 10.6268714
time: 1.66 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8439266, upper bound: 10.8439987
time: 1.86 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -4.3596220, 3.6176953, -4.8939848, 4.0257545, -8.3853760, 8.5116796
1: -3.7762024, 3.4348392, -4.2509437, 3.7665713, -7.5427737, 7.6857829
2: -4.9514799, 3.5895605, -5.5441694, 3.9399633, -8.8914433, 9.1337299
3: -5.3584423, 3.1831036, -5.8212814, 3.2330990, -8.5915403, 9.0043850
4: -5.0857267, 3.7469468, -5.6733103, 4.1140733, -9.1998005, 9.4202557
5: -4.2851748, 3.5885410, -4.7948580, 3.9627540, -8.2479286, 8.3833981
6: -4.1274104, 4.0326872, -4.5430226, 4.4767122, -8.6041222, 8.5757103
7: -4.3985171, 4.2867756, -4.9111648, 4.7876940, -9.1862106, 9.1979399
8: -6.7713509, 3.6085205, -7.5964932, 3.8544226, -10.6257734, 11.2050133
9: -3.8378708, 4.1155744, -4.2639904, 4.5530343, -8.3909035, 8.3795633

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4347758, upper bound: 10.5259260
time: 1.90 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8406201, upper bound: 10.8406201
time: 2.74 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 6.03 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 6.03
Output dim: 8, lower bound: -10.4588523, upper bound: 10.6268714
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 6.03
Output dim: 8, lower bound: -10.8439266, upper bound: 10.8439987
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 6.03
Output dim: 8, lower bound: -10.4347758, upper bound: 10.5259260
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 6.03
Output dim: 8, lower bound: -10.8406201, upper bound: 10.8406201

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -1.0109360, 1.0661721, -2.9171205, 2.4092093, -3.4201453, 3.9832926
1: -0.8210928, 0.9583632, -2.3849864, 2.3493662, -3.1704590, 3.3433495
2: -0.9250798, 1.2859070, -3.1864502, 2.5465593, -3.4716392, 4.4723573
3: -0.9066604, 1.0202264, -3.3806784, 2.1769562, -3.0836167, 4.4009047
4: -1.0571092, 0.9829249, -3.3000588, 2.5169952, -3.5741043, 4.2829838
5: -0.9172652, 1.0346648, -2.7793574, 2.4250696, -3.3423347, 3.8140221
6: -0.9817039, 0.9888433, -2.7171481, 2.6823256, -3.6640296, 3.7059913
7: -0.9717048, 1.0027300, -2.9073620, 2.8174405, -3.7891452, 3.9100919
8: -1.3888791, 2.6229784, -4.5574760, 2.9962656, -4.3851447, 7.1804543
9: -0.9480655, 1.1377145, -2.4900370, 2.7667885, -3.7148540, 3.6277514

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4492942, upper bound: 10.5779446
time: 2.43 seconds

## Relational analysis of IS_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4492942, upper bound: 10.6268714
time: 2.14 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -3.7270885, 3.0890789, -4.2669077, 3.5424657, -7.2695541, 7.3559847
1: -3.1653371, 2.9619009, -3.6857367, 3.3653643, -6.5307016, 6.6476374
2: -4.1831040, 3.1329398, -4.8414631, 3.5199237, -7.7030277, 7.9744029
3: -4.5280991, 2.7820382, -5.2430515, 3.1252191, -7.6533184, 8.0250893
4: -4.3120480, 3.2161894, -4.9768176, 3.6704021, -7.9824500, 8.1930065
5: -3.6234443, 3.0854201, -4.1910505, 3.5149186, -7.1383629, 7.2764707
6: -3.5185735, 3.4486907, -4.0395508, 3.9490709, -7.4676447, 7.4882412
7: -3.7487261, 3.6495347, -4.3064013, 4.1961169, -7.9448433, 7.9559345
8: -5.8063211, 3.2942400, -6.6314263, 3.5575144, -9.3638344, 9.9256668
9: -3.2495794, 3.5275011, -3.7539837, 4.0308790, -7.2804585, 7.2814846

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6902139, upper bound: 10.6706181
time: 2.79 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6902139, upper bound: 10.6706181
time: 3.76 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -0.6166275, 0.7989485, -3.2797279, 2.7439437, -3.3605711, 4.0786762
1: -0.5099363, 0.6488933, -2.7206697, 2.5822659, -3.0922022, 3.3695631
2: -0.5919858, 0.9725808, -3.6041501, 2.8034062, -3.3953919, 4.5767307
3: -0.4385257, 0.7381055, -3.7181687, 2.2313666, -2.6698923, 4.4562740
4: -0.6488171, 0.6450522, -3.7274528, 2.7856221, -3.4344392, 4.3725052
5: -0.5853552, 0.7166393, -3.1374860, 2.6709588, -3.2563138, 3.8541253
6: -0.6016781, 0.6667634, -3.0243926, 3.0063031, -3.6079812, 3.6911559
7: -0.6159548, 0.6319572, -3.2546813, 3.1649461, -3.7809010, 3.8866386
8: -0.6625817, 2.5692225, -5.1391368, 3.1918633, -3.8544450, 7.7083592
9: -0.7083535, 0.8158872, -2.8064964, 3.0744371, -3.7827907, 3.6223836

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_B2_A1_A1

### Relational analysis result of IS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3262735, upper bound: 10.3842496
time: 2.08 seconds

## Relational analysis of IS_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2340819, upper bound: 10.3269938
time: 1.80 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -3.0279815, 2.5012472, -4.4447494, 3.6675324, -6.6955137, 6.9459963
1: -2.4909463, 2.4402728, -3.8298991, 3.4398050, -5.9307513, 6.2701721
2: -3.3267605, 2.6378677, -5.0120091, 3.6230164, -6.9497762, 7.6498766
3: -3.5649118, 2.3094866, -5.2473907, 2.9641776, -6.5290890, 7.5568771
4: -3.4349449, 2.6212492, -5.1350389, 3.7489035, -7.1838484, 7.7562871
5: -2.8936615, 2.5241451, -4.3345184, 3.6049585, -6.4986191, 6.8586636
6: -2.8353622, 2.7921808, -4.1258402, 4.0685873, -6.9039497, 6.9180212
7: -3.0196805, 2.9321327, -4.4470344, 4.3408108, -7.3604913, 7.3791661
8: -4.7286358, 3.0354779, -6.9228234, 3.6378598, -8.3664951, 9.9583015
9: -2.5949152, 2.8791103, -3.8593097, 4.1436839, -6.7385974, 6.7384200

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5259260, upper bound: 10.4347758
time: 2.14 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5259260, upper bound: 10.8406201
time: 1.99 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 5.47 seconds
IS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 5.47
Output dim: 8, lower bound: -10.4492942, upper bound: 10.5779446
IS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 5.47
Output dim: 8, lower bound: -10.4492942, upper bound: 10.6268714
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 5.47
Output dim: 8, lower bound: -10.6902139, upper bound: 10.6706181
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 5.47
Output dim: 8, lower bound: -10.6902139, upper bound: 10.6706181
IS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 5.47
Output dim: 8, lower bound: -10.3262735, upper bound: 10.3842496
IS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 5.47
Output dim: 8, lower bound: -10.2340819, upper bound: 10.3269938
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 5.47
Output dim: 8, lower bound: -10.5259260, upper bound: 10.4347758
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 5.47
Output dim: 8, lower bound: -10.5259260, upper bound: 10.8406201

## BFS IS instance: IS_B1_A1_B1

### Backsubstitution after applying IS history:
0: -1.0109360, 1.0661721, -0.8224976, 0.9526591, -1.9635952, 1.8886697
1: -0.8210928, 0.9583632, -0.6783867, 0.8205066, -1.6415994, 1.6367500
2: -0.9250798, 1.2859070, -0.7437838, 1.1596640, -2.0847437, 2.0296907
3: -0.9066604, 1.0202264, -0.6810015, 0.9041843, -1.8108448, 1.7012279
4: -1.0571092, 0.9829249, -0.8567464, 0.8332369, -1.8903461, 1.8396714
5: -0.9172652, 1.0346648, -0.7553797, 0.8981278, -1.8153930, 1.7900444
6: -0.9817039, 0.9888433, -0.8005887, 0.8345573, -1.8162613, 1.7894320
7: -0.9717048, 1.0027300, -0.8075277, 0.8357326, -1.8074374, 1.8102577
8: -1.3888791, 2.6229784, -1.0656095, 2.6009374, -3.9898164, 3.6885879
9: -0.9480655, 1.1377145, -0.8306347, 0.9904779, -1.9385433, 1.9683492

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4068180, upper bound: 10.5138878
time: 2.77 seconds

## Relational analysis of IS_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2893873, upper bound: 10.4274707
time: 1.90 seconds

## BFS IS instance: IS_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1.0109360, 1.0661721, -3.3821220, 2.7721789, -3.7831149, 4.4482942
1: -0.8210928, 0.9583632, -2.8427665, 2.6969657, -3.5180585, 3.8011298
2: -0.9250798, 1.2859070, -3.7620134, 2.8720245, -3.7971044, 5.0479202
3: -0.9066604, 1.0202264, -4.0491743, 2.5488682, -3.4555287, 5.0694008
4: -1.0571092, 0.9829249, -3.8139873, 2.9232626, -3.9803720, 4.7969122
5: -0.9172652, 1.0346648, -3.2598171, 2.7886219, -3.7058871, 4.2944818
6: -0.9817039, 0.9888433, -3.1692996, 3.1174283, -4.0991321, 4.1581430
7: -0.9717048, 1.0027300, -3.3878736, 3.3019173, -4.2736220, 4.3906035
8: -1.3888791, 2.6229784, -5.2810340, 3.1540082, -4.5428872, 7.9040127
9: -0.9480655, 1.1377145, -2.9147217, 3.1973007, -4.1453662, 4.0524364

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_B1_A1_B2_B1

### Relational analysis result of IS_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2894134, upper bound: 10.4737945
time: 2.18 seconds

## Relational analysis of IS_B1_A1_B2_B2

### Relational analysis result of IS_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2893873, upper bound: 10.4738321
time: 2.92 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -3.7270885, 3.0890789, -0.8224976, 0.9526591, -4.6797476, 3.9115765
1: -3.1653371, 2.9619009, -0.6783867, 0.8205066, -3.9858437, 3.6402876
2: -4.1831040, 3.1329398, -0.7437838, 1.1596640, -5.3427682, 3.8767238
3: -4.5280991, 2.7820382, -0.6810015, 0.9041843, -5.4322834, 3.4630396
4: -4.3120480, 3.2161894, -0.8567464, 0.8332369, -5.1452847, 4.0729361
5: -3.6234443, 3.0854201, -0.7553797, 0.8981278, -4.5215721, 3.8407998
6: -3.5185735, 3.4486907, -0.8005887, 0.8345573, -4.3531308, 4.2492795
7: -3.7487261, 3.6495347, -0.8075277, 0.8357326, -4.5844588, 4.4570622
8: -5.8063211, 3.2942400, -1.0656095, 2.6009374, -8.4072590, 4.3598495
9: -3.2495794, 3.5275011, -0.8306347, 0.9904779, -4.2400575, 4.3581357

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6421804, upper bound: 10.6272372
time: 2.38 seconds

## Relational analysis of IS_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6288355, upper bound: 10.6066275
time: 4.29 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -3.7270885, 3.0890789, -3.4052205, 2.8175251, -6.5446138, 6.4942994
1: -3.1653371, 2.9619009, -2.8558207, 2.7208309, -5.8861680, 5.8177214
2: -4.1831040, 3.1329398, -3.7890821, 2.9046569, -7.0877609, 6.9220219
3: -4.5280991, 2.7820382, -4.0843210, 2.5608428, -7.0889416, 6.8663592
4: -4.3120480, 3.2161894, -3.9089458, 2.9427168, -7.2547646, 7.1251354
5: -3.6234443, 3.0854201, -3.2855368, 2.8261533, -6.4495974, 6.3709569
6: -3.5185735, 3.4486907, -3.2041881, 3.1468105, -6.6653843, 6.6528788
7: -3.7487261, 3.6495347, -3.4125867, 3.3201020, -7.0688281, 7.0621214
8: -5.8063211, 3.2942400, -5.3120508, 3.1667218, -8.9730434, 8.6062908
9: -3.2495794, 3.5275011, -2.9472203, 3.2273698, -6.4769492, 6.4747214

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6421805, upper bound: 10.8415949
time: 2.44 seconds

## Relational analysis of IS_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6288356, upper bound: 10.8377708
time: 2.60 seconds

## BFS IS instance: IS_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.4399655, 0.6378927, -3.0567014, 2.5617790, -3.0017445, 3.6945941
1: -0.3653784, 0.4780297, -2.5034494, 2.4206457, -2.7860241, 2.9814792
2: -0.5046318, 0.7774706, -3.3322976, 2.6494820, -3.1541138, 4.1097679
3: -0.2791503, 0.5823917, -3.4266276, 2.1030025, -2.3821528, 4.0090194
4: -0.4951183, 0.4719853, -3.4517324, 2.5978360, -3.0929544, 3.9237177
5: -0.4196697, 0.5732877, -2.9084036, 2.4962811, -2.9159508, 3.4816914
6: -0.4193625, 0.5160049, -2.8123510, 2.8000989, -3.2194614, 3.3283558
7: -0.4618016, 0.4638760, -3.0223713, 2.9365859, -3.3983874, 3.4862473
8: -0.3309709, 2.4645641, -4.7895365, 3.1105521, -3.4415231, 7.2541008
9: -0.6216207, 0.6466268, -2.6092424, 2.8690686, -3.4906893, 3.2558694

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_B2_A1_A1_B1

### Relational analysis result of IS_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0467749, upper bound: 10.0726482
time: 1.81 seconds

## Relational analysis of IS_B2_A1_A1_B2

### Relational analysis result of IS_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0942922, upper bound: 10.1571122
time: 1.59 seconds

## BFS IS instance: IS_B2_A1_A2

### Backsubstitution after applying IS history:
0: -1.2261815, 1.3062198, -2.9545913, 2.4783230, -3.7045045, 4.2608109
1: -0.9791979, 1.1733789, -2.4030476, 2.3472841, -3.3264818, 3.5764265
2: -0.9199095, 1.6271781, -3.2065151, 2.5789423, -3.4988518, 4.8336930
3: -1.0922583, 1.1746113, -3.2927847, 2.0425429, -3.1348014, 4.4673958
4: -1.1806741, 1.2008911, -3.3244524, 2.5110183, -3.6916924, 4.5253434
5: -1.0925370, 1.1990054, -2.8029633, 2.4170940, -3.5096312, 4.0019684
6: -1.1249063, 1.1679790, -2.7152414, 2.7055042, -3.8304105, 3.8832204
7: -1.1833377, 1.1946260, -2.9153070, 2.8318524, -4.0151901, 4.1099329
8: -1.7498575, 2.5849457, -4.6279354, 3.0671742, -4.8170319, 7.2128811
9: -1.0911226, 1.3417611, -2.5209908, 2.7739432, -3.8650658, 3.8627520

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_B2_A1_A2_B1

### Relational analysis result of IS_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2093583, upper bound: 10.2093583
time: 2.44 seconds

## Relational analysis of IS_B2_A1_A2_B2

### Relational analysis result of IS_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2093583, upper bound: 10.3269938
time: 1.75 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -3.0279815, 2.5012472, -1.2885556, 1.2915748, -4.3195562, 3.7898028
1: -2.4909463, 2.4402728, -0.9978373, 1.1210864, -3.6120327, 3.4381101
2: -3.3267605, 2.6378677, -1.1817958, 1.4661613, -4.7929220, 3.8196635
3: -3.5649118, 2.3094866, -1.2152059, 1.0104641, -4.5753756, 3.5246925
4: -3.4349449, 2.6212492, -1.3690012, 1.1674459, -4.6023908, 3.9902503
5: -2.8936615, 2.5241451, -1.1490538, 1.2066720, -4.1003332, 3.6731989
6: -2.8353622, 2.7921808, -1.1961780, 1.2265388, -4.0619011, 3.9883587
7: -3.0196805, 2.9321327, -1.2147151, 1.2315087, -4.2511892, 4.1468477
8: -4.7286358, 3.0354779, -1.8933592, 2.8310375, -7.5596733, 4.9288368
9: -2.5949152, 2.8791103, -1.1436851, 1.3561463, -3.9510615, 4.0227957

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3679715, upper bound: 10.2743488
time: 1.80 seconds

## Relational analysis of IS_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3639818, upper bound: 10.2732735
time: 2.36 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -3.0279815, 2.5012472, -3.6654465, 3.0519638, -6.0799456, 6.1666927
1: -2.4909463, 2.4402728, -3.0935688, 2.8719342, -5.3628807, 5.5338416
2: -3.3267605, 2.6378677, -4.0813146, 3.0818248, -6.4085855, 6.7191820
3: -3.5649118, 2.3094866, -4.2469306, 2.4954326, -6.0603447, 6.5564175
4: -3.4349449, 2.6212492, -4.2021399, 3.1118362, -6.5467811, 6.8233891
5: -2.8936615, 2.5241451, -3.5362399, 2.9846327, -5.8782940, 6.0603848
6: -2.8353622, 2.7921808, -3.4004350, 3.3648591, -6.2002211, 6.1926155
7: -3.0196805, 2.9321327, -3.6567678, 3.5623693, -6.5820498, 6.5889006
8: -4.7286358, 3.0354779, -5.7447891, 3.3173430, -8.0459785, 8.7802668
9: -2.5949152, 2.8791103, -3.1585369, 3.4368827, -6.0317979, 6.0376472

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3679716, upper bound: 10.8014411
time: 1.90 seconds

## Relational analysis of IS_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3639818, upper bound: 10.8002446
time: 3.27 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 14.70 seconds
IS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 14.70
Output dim: 8, lower bound: -10.4068180, upper bound: 10.5138878
IS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 14.70
Output dim: 8, lower bound: -10.2893873, upper bound: 10.4274707
IS_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 14.70
Output dim: 8, lower bound: -10.2894134, upper bound: 10.4737945
IS_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 14.70
Output dim: 8, lower bound: -10.2893873, upper bound: 10.4738321
IS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 14.70
Output dim: 8, lower bound: -10.6421804, upper bound: 10.6272372
IS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 14.70
Output dim: 8, lower bound: -10.6288355, upper bound: 10.6066275
IS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 14.70
Output dim: 8, lower bound: -10.6421805, upper bound: 10.8415949
IS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 14.70
Output dim: 8, lower bound: -10.6288356, upper bound: 10.8377708
IS_B2_A1_A1_B1, status: Status.VERIFIED, split count: 4, time: 14.70
Output dim: 8, lower bound: -10.0467749, upper bound: 10.0726482
IS_B2_A1_A1_B2, status: Status.VERIFIED, split count: 4, time: 14.70
Output dim: 8, lower bound: -10.0942922, upper bound: 10.1571122
IS_B2_A1_A2_B1, status: Status.VERIFIED, split count: 4, time: 14.70
Output dim: 8, lower bound: -10.2093583, upper bound: 10.2093583
IS_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.70
Output dim: 8, lower bound: -10.2093583, upper bound: 10.3269938
IS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 14.70
Output dim: 8, lower bound: -10.3679715, upper bound: 10.2743488
IS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 14.70
Output dim: 8, lower bound: -10.3639818, upper bound: 10.2732735
IS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 14.70
Output dim: 8, lower bound: -10.3679716, upper bound: 10.8014411
IS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 14.70
Output dim: 8, lower bound: -10.3639818, upper bound: 10.8002446

## BFS IS instance: IS_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.5823044, 0.7623762, -0.6612955, 0.8342103, -1.4165146, 1.4236717
1: -0.4856315, 0.6175168, -0.5503117, 0.6839608, -1.1695924, 1.1678284
2: -0.5682570, 0.9213098, -0.6182889, 1.0145186, -1.5827756, 1.5395987
3: -0.4081050, 0.7298506, -0.4953497, 0.7858531, -1.1939582, 1.2252003
4: -0.6276305, 0.6173378, -0.7009859, 0.6903644, -1.3179948, 1.3183237
5: -0.5609183, 0.6906986, -0.6246184, 0.7620667, -1.3229849, 1.3153170
6: -0.5705864, 0.6374269, -0.6492186, 0.6990575, -1.2696440, 1.2866454
7: -0.5943890, 0.6131480, -0.6666925, 0.6861237, -1.2805127, 1.2798405
8: -0.5885465, 2.5068948, -0.7507786, 2.5520401, -3.1405866, 3.2576735
9: -0.6914485, 0.7806575, -0.7313614, 0.8553208, -1.5467693, 1.5120189

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_B1_A1_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1441971, upper bound: 10.2156970
time: 6.94 seconds

## Relational analysis of IS_B1_A1_B1_A1_B2

### Relational analysis result of IS_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1835947, upper bound: 10.2971485
time: 2.36 seconds

## BFS IS instance: IS_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.9637101, 1.0354692, -0.6780854, 0.8468478, -1.8105578, 1.7135546
1: -0.7814004, 0.9083995, -0.5643547, 0.6980037, -1.4794042, 1.4727542
2: -0.8783789, 1.2366799, -0.6298130, 1.0325100, -1.9108889, 1.8664929
3: -0.8607405, 0.9741442, -0.5138784, 0.7996606, -1.6604011, 1.4880226
4: -1.0190847, 0.9430204, -0.7166800, 0.7061533, -1.7252380, 1.6597004
5: -0.8752270, 0.9973388, -0.6383705, 0.7770494, -1.6522763, 1.6357093
6: -0.9339508, 0.9458307, -0.6658063, 0.7115808, -1.6455317, 1.6116370
7: -0.9377082, 0.9708058, -0.6815936, 0.7025371, -1.6402452, 1.6523993
8: -1.3006881, 2.5960696, -0.7836897, 2.5595200, -3.8602080, 3.3797593
9: -0.9164696, 1.0967474, -0.7409934, 0.8701606, -1.7866302, 1.8377408

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_B1_A1_B1_A2_B1

### Relational analysis result of IS_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0290989, upper bound: 10.1379494
time: 2.10 seconds

## Relational analysis of IS_B1_A1_B1_A2_B2

### Relational analysis result of IS_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0496060, upper bound: 10.2000592
time: 2.09 seconds

## BFS IS instance: IS_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.7887654, 0.9257174, -2.6256793, 2.1719642, -2.9607296, 3.5513966
1: -0.6532334, 0.7889489, -2.1261425, 2.1281936, -2.7814269, 2.9150915
2: -0.7158680, 1.1270115, -2.8324234, 2.3360937, -3.0519617, 3.9594350
3: -0.6480175, 0.8900136, -3.0545502, 2.0663738, -2.7143912, 3.9445639
4: -0.8309270, 0.8052841, -2.9825959, 2.2827604, -3.1136873, 3.7878799
5: -0.7283072, 0.8741019, -2.4819634, 2.2193336, -2.9476409, 3.3560653
6: -0.7714838, 0.8040364, -2.4621646, 2.4203858, -3.1918697, 3.2662010
7: -0.7835889, 0.8117791, -2.6212811, 2.5360684, -3.3196573, 3.4330602
8: -1.0010104, 2.5708289, -4.0910535, 2.8534055, -3.8544159, 6.6618824
9: -0.8102062, 0.9621032, -2.2357306, 2.5189247, -3.3291309, 3.1978340

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_B1_A1_B2_B1_A1

### Relational analysis result of IS_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2998031, upper bound: 10.4737413
time: 2.71 seconds

## Relational analysis of IS_B1_A1_B2_B1_A2

### Relational analysis result of IS_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2998031, upper bound: 10.4737413
time: 2.63 seconds

## BFS IS instance: IS_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.8113918, 0.9408661, -3.2581463, 2.6739514, -3.4853432, 4.1990123
1: -0.6715765, 0.8060303, -2.7026410, 2.5836129, -3.2551894, 3.5086713
2: -0.7354344, 1.1455388, -3.6069539, 2.7651370, -3.5005713, 4.7524929
3: -0.6744518, 0.9045583, -3.8830571, 2.4436963, -3.1181481, 4.7876153
4: -0.8525327, 0.8247910, -3.6885266, 2.8131232, -3.6656559, 4.5133176
5: -0.7470209, 0.8913012, -3.1369815, 2.6878891, -3.4349101, 4.0282826
6: -0.7924621, 0.8231571, -3.0517068, 2.9951763, -3.7876384, 3.8748639
7: -0.8026726, 0.8314747, -3.2679448, 3.1787872, -3.9814599, 4.0994196
8: -1.0425594, 2.5784807, -5.0800896, 3.0789616, -4.1215210, 7.6585703
9: -0.8232400, 0.9810954, -2.8009467, 3.0785897, -3.9018297, 3.7820420

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_B1_A1_B2_B2_A1

### Relational analysis result of IS_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2998031, upper bound: 10.4738320
time: 2.15 seconds

## Relational analysis of IS_B1_A1_B2_B2_A2

### Relational analysis result of IS_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2998031, upper bound: 10.4738320
time: 2.20 seconds

## BFS IS instance: IS_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -2.9328561, 2.4127960, -0.6612955, 0.8342103, -3.7670665, 3.0740914
1: -2.3821862, 2.3555315, -0.5503117, 0.6839608, -3.0661469, 2.9058433
2: -3.2099998, 2.5388985, -0.6182889, 1.0145186, -4.2245183, 3.1571875
3: -3.4837213, 2.2754459, -0.4953497, 0.7858531, -4.2695742, 2.7707956
4: -3.3391724, 2.5442429, -0.7009859, 0.6903644, -4.0295367, 3.2452288
5: -2.8007469, 2.4521141, -0.6246184, 0.7620667, -3.5628135, 3.0767326
6: -2.7534752, 2.7012711, -0.6492186, 0.6990575, -3.4525328, 3.3504896
7: -2.9355774, 2.8502209, -0.6666925, 0.6861237, -3.6217012, 3.5169134
8: -4.5668526, 2.9443765, -0.7507786, 2.5520401, -7.1188927, 3.6951551
9: -2.5102952, 2.7934465, -0.7313614, 0.8553208, -3.3656158, 3.5248079

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_B1_A2_B1_A1_A1

### Relational analysis result of IS_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3330525, upper bound: 10.3526300
time: 1.90 seconds

## Relational analysis of IS_B1_A2_B1_A1_A2

### Relational analysis result of IS_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4414304, upper bound: 10.4247140
time: 2.70 seconds

## BFS IS instance: IS_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -3.6088955, 2.9799416, -0.6780854, 0.8468478, -4.4557433, 3.6580272
1: -3.0334468, 2.8518283, -0.5643547, 0.6980037, -3.7314506, 3.4161830
2: -4.0335865, 3.0179672, -0.6298130, 1.0325100, -5.0660963, 3.6477802
3: -4.3737674, 2.6712751, -0.5138784, 0.7996606, -5.1734281, 3.1851535
4: -4.1703949, 3.1077013, -0.7166800, 0.7061533, -4.8765483, 3.8243814
5: -3.5007262, 2.9792786, -0.6383705, 0.7770494, -4.2777758, 3.6176491
6: -3.3946471, 3.3272810, -0.6658063, 0.7115808, -4.1062279, 3.9930873
7: -3.6294036, 3.5355167, -0.6815936, 0.7025371, -4.3319407, 4.2171102
8: -5.6099930, 3.2119558, -0.7836897, 2.5595200, -8.1695127, 3.9956455
9: -3.1330554, 3.4108317, -0.7409934, 0.8701606, -4.0032158, 4.1518250

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_B1_A2_B1_A2_A1

### Relational analysis result of IS_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3131469, upper bound: 10.3267432
time: 2.05 seconds

## Relational analysis of IS_B1_A2_B1_A2_A2

### Relational analysis result of IS_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4279370, upper bound: 10.4030513
time: 2.60 seconds

## BFS IS instance: IS_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -2.9328561, 2.4127960, -3.0883577, 2.5467279, -5.4795837, 5.5011539
1: -2.3821862, 2.3555315, -2.5388188, 2.4750776, -4.8572636, 4.8943501
2: -3.2099998, 2.5388985, -3.4013002, 2.6653361, -5.8753357, 5.9401989
3: -3.4837213, 2.2754459, -3.6682503, 2.3589551, -5.8426762, 5.9436960
4: -3.3391724, 2.5442429, -3.5202122, 2.6734188, -6.0125914, 6.0644550
5: -2.8007469, 2.4521141, -2.9583192, 2.5731437, -5.3738909, 5.4104333
6: -2.7534752, 2.7012711, -2.8972917, 2.8470900, -5.6005650, 5.5985627
7: -2.9355774, 2.8502209, -3.0888228, 3.0008030, -5.9363804, 5.9390440
8: -4.5668526, 2.9443765, -4.8144937, 3.0287681, -7.5956206, 7.7588701
9: -2.5102952, 2.7934465, -2.6515465, 2.9349995, -5.4452944, 5.4449930

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_B1_A2_B2_A1_B1

### Relational analysis result of IS_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8089276, upper bound: 10.8377654
time: 2.32 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2

### Relational analysis result of IS_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8089276, upper bound: 10.8377707
time: 3.35 seconds

## BFS IS instance: IS_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -3.6088955, 2.9799416, -3.1183994, 2.5732698, -6.1821651, 6.0983410
1: -3.0334468, 2.8518283, -2.5712657, 2.5010595, -5.5345063, 5.4230938
2: -4.0335865, 3.0179672, -3.4382429, 2.6907985, -6.7243853, 6.4562101
3: -4.3737674, 2.6712751, -3.7073720, 2.3794723, -6.7532396, 6.3786469
4: -4.1703949, 3.1077013, -3.5562634, 2.6996932, -6.8700881, 6.6639647
5: -3.5007262, 2.9792786, -2.9893138, 2.5976183, -6.0983448, 5.9685926
6: -3.3946471, 3.3272810, -2.9270360, 2.8762727, -6.2709198, 6.2543173
7: -3.6294036, 3.5355167, -3.1186116, 3.0306363, -6.6600399, 6.6541281
8: -5.6099930, 3.2119558, -4.8624253, 3.0457692, -8.6557617, 8.0743809
9: -3.1330554, 3.4108317, -2.6803725, 2.9632652, -6.0963206, 6.0912042

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_B1_A2_B2_A2_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8089276, upper bound: 10.8377654
time: 2.64 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2

### Relational analysis result of IS_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8089276, upper bound: 10.8377708
time: 2.81 seconds

## BFS IS instance: IS_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -1.2261815, 1.3062198, -3.3253064, 2.7745113, -4.0006928, 4.6315260
1: -0.9791979, 1.1733789, -2.7683277, 2.6246052, -3.6038032, 3.9417067
2: -0.9199095, 1.6271781, -3.6687129, 2.8438945, -3.7638040, 5.2958908
3: -1.0922583, 1.1746113, -3.8050508, 2.3003664, -3.3926249, 4.9796619
4: -1.1806741, 1.2008911, -3.7885990, 2.8257647, -4.0064387, 4.9894900
5: -1.0925370, 1.1990054, -3.1907544, 2.7119479, -3.8044848, 4.3897600
6: -1.1249063, 1.1679790, -3.0704439, 3.0525312, -4.1774378, 4.2384229
7: -1.1833377, 1.1946260, -3.3012130, 3.2198911, -4.4032288, 4.4958391
8: -1.7498575, 2.5849457, -5.2129774, 3.1725335, -4.9223909, 7.7979231
9: -1.0911226, 1.3417611, -2.8529987, 3.1251621, -4.2162848, 4.1947598

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_B2_A1_A2_B2_B1

### Relational analysis result of IS_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2093583, upper bound: 10.3269938
time: 1.66 seconds

## Relational analysis of IS_B2_A1_A2_B2_B2

### Relational analysis result of IS_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2093583, upper bound: 10.3269938
time: 2.17 seconds

## BFS IS instance: IS_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -2.2877104, 1.9087899, -1.0731872, 1.1607205, -3.4484310, 2.9819770
1: -1.8476956, 1.8796308, -0.8373573, 0.9601887, -2.8078842, 2.7169881
2: -2.4105849, 2.1123986, -0.9714405, 1.3179774, -3.7285624, 3.0838392
3: -2.5813680, 1.8356632, -0.9692649, 0.9019313, -3.4832993, 2.8049281
4: -2.5926938, 1.9941437, -1.1269585, 1.0061640, -3.5988579, 3.1211023
5: -2.1247373, 1.9644263, -0.9575926, 1.0531656, -3.1779027, 2.9220190
6: -2.1501877, 2.1079900, -1.0043336, 1.0486889, -3.1988766, 3.1123238
7: -2.2694120, 2.1918299, -1.0183607, 1.0607555, -3.3301675, 3.2101908
8: -3.5469000, 2.7682288, -1.5200527, 2.7787313, -6.3256311, 4.2882814
9: -1.9375224, 2.2168264, -0.9959816, 1.1858504, -3.1233728, 3.2128081

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_B2_A2_B1_A1_B1

### Relational analysis result of IS_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3639818, upper bound: 10.2732735
time: 2.11 seconds

## Relational analysis of IS_B2_A2_B1_A1_B2

### Relational analysis result of IS_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3639818, upper bound: 10.2732735
time: 2.19 seconds

## BFS IS instance: IS_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -2.9069414, 2.3972228, -1.0983660, 1.1752684, -4.0822096, 3.4955888
1: -2.3556957, 2.3274298, -0.8553005, 0.9820984, -3.3377941, 3.1827302
2: -3.1762242, 2.5260589, -0.9955319, 1.3360803, -4.5123043, 3.5215907
3: -3.4031565, 2.2061355, -0.9972059, 0.9149535, -4.3181100, 3.2033415
4: -3.2982602, 2.5139987, -1.1532403, 1.0256796, -4.3239398, 3.6672392
5: -2.7725687, 2.4218564, -0.9797614, 1.0711894, -3.8437581, 3.4016178
6: -2.7182167, 2.6712651, -1.0268713, 1.0697291, -3.7879457, 3.6981363
7: -2.9044781, 2.8140368, -1.0407318, 1.0799347, -3.9844127, 3.8547688
8: -4.5336490, 2.9664898, -1.5650713, 2.7871346, -7.3207836, 4.5315609
9: -2.4802496, 2.7612855, -1.0132402, 1.2059033, -3.6861529, 3.7745256

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_B2_A2_B1_A2_B1

### Relational analysis result of IS_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3639818, upper bound: 10.2732735
time: 1.96 seconds

## Relational analysis of IS_B2_A2_B1_A2_B2

### Relational analysis result of IS_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3639818, upper bound: 10.2732735
time: 2.06 seconds

## BFS IS instance: IS_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -2.2877104, 1.9087899, -3.3875391, 2.8261530, -5.1138635, 5.2963290
1: -1.8476956, 1.8796308, -2.8144519, 2.6556194, -4.5033150, 4.6940827
2: -2.4105849, 2.1123986, -3.7427049, 2.8758640, -5.2864490, 5.8551035
3: -2.5813680, 1.8356632, -3.8901303, 2.3231845, -4.9045525, 5.7257934
4: -2.5926938, 1.9941437, -3.8720617, 2.8785496, -5.4712434, 5.8662052
5: -2.1247373, 1.9644263, -3.2553036, 2.7616289, -4.8863659, 5.2197299
6: -2.1501877, 2.1079900, -3.1378045, 3.1051850, -5.2553730, 5.2457943
7: -2.2694120, 2.1918299, -3.3748651, 3.2849381, -5.5543499, 5.5666952
8: -3.5469000, 2.7682288, -5.3076253, 3.1945670, -6.7414670, 8.0758543
9: -1.9375224, 2.2168264, -2.9076352, 3.1786747, -5.1161971, 5.1244617

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_B2_A2_B2_A1_B1

### Relational analysis result of IS_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8002439, upper bound: 10.8002445
time: 1.56 seconds

## Relational analysis of IS_B2_A2_B2_A1_B2

### Relational analysis result of IS_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8002439, upper bound: 10.8002445
time: 1.71 seconds

## BFS IS instance: IS_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -2.9069414, 2.3972228, -3.4131951, 2.8467779, -5.7537193, 5.8104181
1: -2.3556957, 2.3274298, -2.8427856, 2.6781769, -5.0338726, 5.1702156
2: -3.1762242, 2.5260589, -3.7738323, 2.8958693, -6.0720935, 6.2998915
3: -3.4031565, 2.2061355, -3.9218144, 2.3386483, -5.7418051, 6.1279497
4: -3.2982602, 2.5139987, -3.9016438, 2.9006772, -6.1989374, 6.4156427
5: -2.7725687, 2.4218564, -3.2805617, 2.7825580, -5.5551267, 5.7024183
6: -2.7182167, 2.6712651, -3.1618438, 3.1299157, -5.8481321, 5.8331089
7: -2.9044781, 2.8140368, -3.4003630, 3.3101737, -6.2146521, 6.2143998
8: -4.5336490, 2.9664898, -5.3490891, 3.2094810, -7.7431297, 8.3155785
9: -2.4802496, 2.7612855, -2.9307275, 3.2028346, -5.6830845, 5.6920128

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_B2_A2_B2_A2_B1

### Relational analysis result of IS_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8002439, upper bound: 10.8002445
time: 1.76 seconds

## Relational analysis of IS_B2_A2_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8002439, upper bound: 10.8002445
time: 3.13 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 6.31 seconds
IS_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 5, time: 6.31
Output dim: 8, lower bound: -10.1441971, upper bound: 10.2156970
IS_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 5, time: 6.31
Output dim: 8, lower bound: -10.1835947, upper bound: 10.2971485
IS_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 5, time: 6.31
Output dim: 8, lower bound: -10.0290989, upper bound: 10.1379494
IS_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 5, time: 6.31
Output dim: 8, lower bound: -10.0496060, upper bound: 10.2000592
IS_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 8, lower bound: -10.2998031, upper bound: 10.4737413
IS_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 8, lower bound: -10.2998031, upper bound: 10.4737413
IS_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 8, lower bound: -10.2998031, upper bound: 10.4738320
IS_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 8, lower bound: -10.2998031, upper bound: 10.4738320
IS_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 8, lower bound: -10.3330525, upper bound: 10.3526300
IS_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 8, lower bound: -10.4414304, upper bound: 10.4247140
IS_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 8, lower bound: -10.3131469, upper bound: 10.3267432
IS_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 8, lower bound: -10.4279370, upper bound: 10.4030513
IS_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 8, lower bound: -10.8089276, upper bound: 10.8377654
IS_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 8, lower bound: -10.8089276, upper bound: 10.8377707
IS_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 8, lower bound: -10.8089276, upper bound: 10.8377654
IS_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 8, lower bound: -10.8089276, upper bound: 10.8377708
IS_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 8, lower bound: -10.2093583, upper bound: 10.3269938
IS_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 8, lower bound: -10.2093583, upper bound: 10.3269938
IS_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 8, lower bound: -10.3639818, upper bound: 10.2732735
IS_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 8, lower bound: -10.3639818, upper bound: 10.2732735
IS_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 8, lower bound: -10.3639818, upper bound: 10.2732735
IS_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 8, lower bound: -10.3639818, upper bound: 10.2732735
IS_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 8, lower bound: -10.8002439, upper bound: 10.8002445
IS_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 8, lower bound: -10.8002439, upper bound: 10.8002445
IS_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 8, lower bound: -10.8002439, upper bound: 10.8002445
IS_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 8, lower bound: -10.8002439, upper bound: 10.8002445

## BFS IS instance: IS_B1_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.5823044, 0.7623762, -2.6256793, 2.1719642, -2.7542686, 3.3880553
1: -0.4856315, 0.6175168, -2.1261425, 2.1281936, -2.6138251, 2.7436593
2: -0.5682570, 0.9213098, -2.8324234, 2.3360937, -2.9043508, 3.7537332
3: -0.4081050, 0.7298506, -3.0545502, 2.0663738, -2.4744787, 3.7844007
4: -0.6276305, 0.6173378, -2.9825959, 2.2827604, -2.9103909, 3.5999336
5: -0.5609183, 0.6906986, -2.4819634, 2.2193336, -2.7802520, 3.1726620
6: -0.5705864, 0.6374269, -2.4621646, 2.4203858, -2.9909723, 3.0995915
7: -0.5943890, 0.6131480, -2.6212811, 2.5360684, -3.1304574, 3.2344291
8: -0.5885465, 2.5068948, -4.0910535, 2.8534055, -3.4419520, 6.5979481
9: -0.6914485, 0.7806575, -2.2357306, 2.5189247, -3.2103732, 3.0163882

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_B1_A1_B2_B1_A1_B1

### Relational analysis result of IS_B1_A1_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0379905, upper bound: 10.1752205
time: 1.58 seconds

## Relational analysis of IS_B1_A1_B2_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0622791, upper bound: 10.2518600
time: 1.98 seconds

## BFS IS instance: IS_B1_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.9398836, 1.0319059, -2.6256793, 2.1719642, -3.1118479, 3.6575851
1: -0.7659042, 0.8978448, -2.1261425, 2.1281936, -2.8940978, 3.0239873
2: -0.8471785, 1.2306426, -2.8324234, 2.3360937, -3.1832721, 4.0630660
3: -0.8361225, 0.9697104, -3.0545502, 2.0663738, -2.9024963, 4.0242605
4: -0.9810215, 0.9360934, -2.9825959, 2.2827604, -3.2637818, 3.9186893
5: -0.8448707, 0.9894065, -2.4819634, 2.2193336, -3.0642045, 3.4713700
6: -0.9088482, 0.9264999, -2.4621646, 2.4203858, -3.3292341, 3.3886645
7: -0.9186203, 0.9539633, -2.6212811, 2.5360684, -3.4546888, 3.5752444
8: -1.2739576, 2.5951953, -4.0910535, 2.8534055, -4.1273632, 6.6862488
9: -0.9020308, 1.0831215, -2.2357306, 2.5189247, -3.4209557, 3.3188522

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_B1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_B1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_B1_A1_B2_B1_A2_A1

### Relational analysis result of IS_B1_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2176350, upper bound: 10.3236621
time: 2.54 seconds

## Relational analysis of IS_B1_A1_B2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_B1_A1_B2_B1_A2_B1

### Relational analysis result of IS_B1_A1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1325542, upper bound: 10.3064628
time: 3.54 seconds

## Relational analysis of IS_B1_A1_B2_B1_A2_B2

### Relational analysis result of IS_B1_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.1340738, upper bound: 10.3132061
time: 2.47 seconds

## BFS IS instance: IS_B1_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.5823044, 0.7623762, -3.2581463, 2.6739514, -3.2562559, 4.0205226
1: -0.4856315, 0.6175168, -2.7026410, 2.5836129, -3.0692444, 3.3201578
2: -0.5682570, 0.9213098, -3.6069539, 2.7651370, -3.3333941, 4.5282636
3: -0.4081050, 0.7298506, -3.8830571, 2.4436963, -2.8518014, 4.6129079
4: -0.6276305, 0.6173378, -3.6885266, 2.8131232, -3.4407537, 4.3058643
5: -0.5609183, 0.6906986, -3.1369815, 2.6878891, -3.2488074, 3.8276801
6: -0.5705864, 0.6374269, -3.0517068, 2.9951763, -3.5657628, 3.6891336
7: -0.5943890, 0.6131480, -3.2679448, 3.1787872, -3.7731762, 3.8810928
8: -0.5885465, 2.5068948, -5.0800896, 3.0789616, -3.6675081, 7.5869846
9: -0.6914485, 0.7806575, -2.8009467, 3.0785897, -3.7700381, 3.5816042

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_B1_A1_B2_B2_A1_B1

### Relational analysis result of IS_B1_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0376586, upper bound: 10.1745094
time: 2.33 seconds

## Relational analysis of IS_B1_A1_B2_B2_A1_B2

### Relational analysis result of IS_B1_A1_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0622790, upper bound: 10.2521523
time: 1.82 seconds

## BFS IS instance: IS_B1_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.9421180, 1.0320566, -3.2581463, 2.6739514, -3.6160693, 4.2902031
1: -0.7760738, 0.8978448, -2.7026410, 2.5836129, -3.3596866, 3.6004858
2: -0.8541905, 1.2306426, -3.6069539, 2.7651370, -3.6193275, 4.8375964
3: -0.8372067, 0.9717373, -3.8830571, 2.4436963, -3.2809029, 4.8547945
4: -0.9941624, 0.9360934, -3.6885266, 2.8131232, -3.8072855, 4.6246200
5: -0.8557203, 0.9894065, -3.1369815, 2.6878891, -3.5436094, 4.1263881
6: -0.9100110, 0.9366938, -3.0517068, 2.9951763, -3.9051874, 3.9884007
7: -0.9200113, 0.9590390, -3.2679448, 3.1787872, -4.0987988, 4.2269840
8: -1.2825596, 2.5954096, -5.0800896, 3.0789616, -4.3615212, 7.6754990
9: -0.9054558, 1.0903533, -2.8009467, 3.0785897, -3.9840455, 3.8913000

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_B1_A1_B2_B2_A2_A1

### Relational analysis result of IS_B1_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2143202, upper bound: 10.3190469
time: 2.41 seconds

## Relational analysis of IS_B1_A1_B2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_B1_A1_B2_B2_A2_A1

### Relational analysis result of IS_B1_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2998031, upper bound: 10.4735759
time: 1.84 seconds

## Relational analysis of IS_B1_A1_B2_B2_A2_A2

### Relational analysis result of IS_B1_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2998031, upper bound: 10.4737413
time: 2.60 seconds

## BFS IS instance: IS_B1_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.5681466, 0.7484194, -0.3530665, 0.5474774, -1.1156240, 1.1014860
1: -0.5009052, 0.6149521, -0.2814330, 0.3940346, -0.8949398, 0.8963851
2: -0.5307309, 0.8295106, -0.4576199, 0.5924907, -1.1232216, 1.2871305
3: -0.4029301, 0.7945162, -0.2134947, 0.4570267, -0.8599569, 1.0080109
4: -0.6048436, 0.6257648, -0.4229846, 0.3691886, -0.9740322, 1.0487494
5: -0.5558202, 0.6852477, -0.3434045, 0.4702854, -1.0261056, 1.0286522
6: -0.5501204, 0.6316398, -0.3112675, 0.4268531, -0.9769734, 0.9429073
7: -0.5907705, 0.6063630, -0.3910034, 0.3669192, -0.9576896, 0.9973664
8: -0.5369560, 2.4385779, -0.0828341, 2.4528990, -2.9898548, 2.5214119
9: -0.6703390, 0.7592965, -0.5785313, 0.5378324, -1.2081714, 1.3378279

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_B1_A2_B1_A1_A1_A1

### Relational analysis result of IS_B1_A2_B1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -9.9099598, upper bound: 9.8949006
time: 2.62 seconds

## Relational analysis of IS_B1_A2_B1_A1_A1_A2

### Relational analysis result of IS_B1_A2_B1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -9.6570093, upper bound: 9.6246803
time: 2.31 seconds

## BFS IS instance: IS_B1_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -1.8043153, 1.5551851, -0.4838380, 0.6839600, -2.4882753, 2.0390229
1: -1.4490350, 1.5350487, -0.4027784, 0.5167377, -1.9657727, 1.9378271
2: -1.7930186, 1.7759449, -0.5273594, 0.8013871, -2.5944057, 2.3033042
3: -1.9157420, 1.5462267, -0.3098397, 0.6330364, -2.5487785, 1.8560665
4: -2.0220578, 1.6002737, -0.5277487, 0.5282099, -2.5502677, 2.1280224
5: -1.6570227, 1.6066689, -0.4677826, 0.6096593, -2.2666821, 2.0744514
6: -1.7033873, 1.6698742, -0.4667977, 0.5552092, -2.2585964, 2.1366720
7: -1.7596881, 1.7187676, -0.5046337, 0.5103711, -2.2700593, 2.2234013
8: -2.7422221, 2.6589584, -0.4034616, 2.5158687, -5.2580910, 3.0624199
9: -1.5202951, 1.7920907, -0.6481612, 0.6853438, -2.2056389, 2.4402518

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_B1_A2_B1_A1_A2_A1

### Relational analysis result of IS_B1_A2_B1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0143492, upper bound: 9.9681528
time: 1.81 seconds

## Relational analysis of IS_B1_A2_B1_A1_A2_A2

### Relational analysis result of IS_B1_A2_B1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -9.7670893, upper bound: 9.7298962
time: 4.63 seconds

## BFS IS instance: IS_B1_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.9397333, 1.0135541, -0.3592169, 0.5538517, -1.4935851, 1.3727710
1: -0.7848649, 0.9064604, -0.2877711, 0.3992085, -1.1840734, 1.1942315
2: -0.8320580, 1.1362298, -0.4607743, 0.6035067, -1.4355646, 1.5970041
3: -0.8558954, 1.0292697, -0.2169175, 0.4671372, -1.3230326, 1.2461872
4: -0.9955692, 0.9431753, -0.4278531, 0.3770877, -1.3726569, 1.3710284
5: -0.8706967, 0.9766896, -0.3492087, 0.4764962, -1.3471929, 1.3258983
6: -0.8978168, 0.9306795, -0.3176564, 0.4337308, -1.3315476, 1.2483358
7: -0.9297712, 0.9544949, -0.3960524, 0.3742004, -1.3039716, 1.3505473
8: -1.2426062, 2.5246916, -0.1007057, 2.4595163, -3.7021224, 2.6253972
9: -0.8968986, 1.0774174, -0.5817349, 0.5451272, -1.4420257, 1.6591523

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_B1_A2_B1_A2_A1_A1

### Relational analysis result of IS_B1_A2_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -9.8736466, upper bound: 9.8460038
time: 5.93 seconds

## Relational analysis of IS_B1_A2_B1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_B1_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_B1_A2_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1972492, upper bound: 10.2144567
time: 2.73 seconds

## Relational analysis of IS_B1_A2_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1835089, upper bound: 10.1964884
time: 2.25 seconds

## BFS IS instance: IS_B1_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -2.3790596, 1.9928509, -0.4954976, 0.6949489, -3.0740085, 2.4883485
1: -1.9283799, 1.9499545, -0.4139854, 0.5292872, -2.4576671, 2.3639398
2: -2.5205500, 2.1615653, -0.5327615, 0.8181286, -3.3386786, 2.6943269
3: -2.7045591, 1.8751864, -0.3199161, 0.6444191, -3.3489783, 2.1951025
4: -2.7054448, 2.0727735, -0.5378658, 0.5396143, -3.2450590, 2.6106391
5: -2.2313783, 2.0340221, -0.4790093, 0.6195303, -2.8509088, 2.5130315
6: -2.2252462, 2.1977654, -0.4804957, 0.5656820, -2.7909281, 2.6782610
7: -2.3680527, 2.2879741, -0.5144389, 0.5211884, -2.8892412, 2.8024130
8: -3.7105646, 2.8064680, -0.4280767, 2.5231318, -6.2336965, 3.2345448
9: -2.0230145, 2.3016105, -0.6532571, 0.6978922, -2.7209067, 2.9548676

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_B1_A2_B1_A2_A2_A1

### Relational analysis result of IS_B1_A2_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -9.9733021, upper bound: 9.9212605
time: 2.35 seconds

## Relational analysis of IS_B1_A2_B1_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_A2_B1_A2_A2_A1

### Relational analysis result of IS_B1_A2_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1688281, upper bound: 10.1393709
time: 2.68 seconds

## Relational analysis of IS_B1_A2_B1_A2_A2_A2

### Relational analysis result of IS_B1_A2_B1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1385483, upper bound: 10.1150341
time: 2.95 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -2.9328561, 2.4127960, -2.6259260, 2.1722479, -5.1051040, 5.0387220
1: -2.3821862, 2.3555315, -2.1264014, 2.1283784, -4.5105648, 4.4819326
2: -3.2099998, 2.5388985, -2.8327496, 2.3362224, -5.5462222, 5.3716478
3: -3.4837213, 2.2754459, -3.0550425, 2.0664408, -5.5501623, 5.3304882
4: -3.3391724, 2.5442429, -2.9829547, 2.2829671, -5.6221395, 5.5271978
5: -2.8007469, 2.4521141, -2.4822688, 2.2194552, -5.0202022, 4.9343829
6: -2.7534752, 2.7012711, -2.4624128, 2.4206142, -5.1740894, 5.1636839
7: -2.9355774, 2.8502209, -2.6215091, 2.5364044, -5.4719820, 5.4717302
8: -4.5668526, 2.9443765, -4.0913000, 2.8535833, -7.4204359, 7.0356765
9: -2.5102952, 2.7934465, -2.2360466, 2.5191741, -5.0294695, 5.0294933

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_B1_A2_B2_A1_B1_B1

### Relational analysis result of IS_B1_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7422154, upper bound: 10.8245306
time: 2.96 seconds

## Relational analysis of IS_B1_A2_B2_A1_B1_B2

### Relational analysis result of IS_B1_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7405626, upper bound: 10.8241305
time: 2.68 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -2.9328561, 2.4127960, -3.2755766, 2.6998014, -5.6326575, 5.6883726
1: -2.3821862, 2.3555315, -2.7131920, 2.6016662, -4.9838524, 5.0687237
2: -3.2099998, 2.5388985, -3.6265893, 2.7814317, -5.9914312, 6.1654878
3: -3.4837213, 2.2754459, -3.9138446, 2.4478853, -5.9316063, 6.1892905
4: -3.3391724, 2.5442429, -3.7541409, 2.8261204, -6.1652927, 6.2983837
5: -2.8007469, 2.4521141, -3.1519122, 2.7134638, -5.5142107, 5.6040263
6: -2.7534752, 2.7012711, -3.0710707, 3.0156674, -5.7691426, 5.7723417
7: -2.9355774, 2.8502209, -3.2830110, 3.1946421, -6.1302195, 6.1332321
8: -4.5668526, 2.9443765, -5.0991282, 3.0879221, -7.6547747, 8.0435047
9: -2.5102952, 2.7934465, -2.8212910, 3.1019900, -5.6122851, 5.6147375

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_B1_A2_B2_A1_B2_B1

### Relational analysis result of IS_B1_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7422154, upper bound: 10.8245307
time: 3.73 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2_B2

### Relational analysis result of IS_B1_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7405626, upper bound: 10.8241305
time: 2.50 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -3.6088955, 2.9799416, -2.6259260, 2.1722479, -5.7811432, 5.6058674
1: -3.0334468, 2.8518283, -2.1264014, 2.1283784, -5.1618252, 4.9782295
2: -4.0335865, 3.0179672, -2.8327496, 2.3362224, -6.3698092, 5.8507166
3: -4.3737674, 2.6712751, -3.0550425, 2.0664408, -6.4402084, 5.7263174
4: -4.1703949, 3.1077013, -2.9829547, 2.2829671, -6.4533620, 6.0906563
5: -3.5007262, 2.9792786, -2.4822688, 2.2194552, -5.7201815, 5.4615474
6: -3.3946471, 3.3272810, -2.4624128, 2.4206142, -5.8152614, 5.7896938
7: -3.6294036, 3.5355167, -2.6215091, 2.5364044, -6.1658077, 6.1570258
8: -5.6099930, 3.2119558, -4.0913000, 2.8535833, -8.4635763, 7.3032560
9: -3.1330554, 3.4108317, -2.2360466, 2.5191741, -5.6522293, 5.6468782

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_B1_A2_B2_A2_B1_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7179373, upper bound: 10.8199559
time: 2.35 seconds

## Relational analysis of IS_B1_A2_B2_A2_B1_B2

### Relational analysis result of IS_B1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7164918, upper bound: 10.8194116
time: 2.25 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -3.6088955, 2.9799416, -3.2755766, 2.6998014, -6.3086967, 6.2555180
1: -3.0334468, 2.8518283, -2.7131920, 2.6016662, -5.6351128, 5.5650206
2: -4.0335865, 3.0179672, -3.6265893, 2.7814317, -6.8150182, 6.6445565
3: -4.3737674, 2.6712751, -3.9138446, 2.4478853, -6.8216524, 6.5851197
4: -4.1703949, 3.1077013, -3.7541409, 2.8261204, -6.9965153, 6.8618422
5: -3.5007262, 2.9792786, -3.1519122, 2.7134638, -6.2141900, 6.1311908
6: -3.3946471, 3.3272810, -3.0710707, 3.0156674, -6.4103146, 6.3983517
7: -3.6294036, 3.5355167, -3.2830110, 3.1946421, -6.8240457, 6.8185277
8: -5.6099930, 3.2119558, -5.0991282, 3.0879221, -8.6979151, 8.3110838
9: -3.1330554, 3.4108317, -2.8212910, 3.1019900, -6.2350454, 6.2321224

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8089276, upper bound: 10.8377654
time: 2.61 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8089027, upper bound: 10.8377565
time: 2.80 seconds

## BFS IS instance: IS_B2_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -1.2261815, 1.3062198, -2.9841256, 2.4975467, -3.7237282, 4.2903452
1: -0.9791979, 1.1733789, -2.4317844, 2.3783789, -3.3575768, 3.6051633
2: -0.9199095, 1.6271781, -3.2512221, 2.6125007, -3.5324101, 4.8784003
3: -1.0922583, 1.1746113, -3.3600059, 2.1035519, -3.1958103, 4.5346174
4: -1.1806741, 1.2008911, -3.3625844, 2.5391955, -3.7198696, 4.5634756
5: -1.0925370, 1.1990054, -2.8358202, 2.4502070, -3.5427442, 4.0348253
6: -1.1249063, 1.1679790, -2.7543998, 2.7360237, -3.8609300, 3.9223788
7: -1.1833377, 1.1946260, -2.9482336, 2.8677421, -4.0510798, 4.1428595
8: -1.7498575, 2.5849457, -4.6782603, 3.0512009, -4.8010583, 7.2632060
9: -1.0911226, 1.3417611, -2.5482907, 2.8110876, -3.9022102, 3.8900518

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_B2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_B2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_B2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_B2_A1_A2_B2_B1_A1

### Relational analysis result of IS_B2_A1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0765472, upper bound: 10.1218231
time: 1.99 seconds

## Relational analysis of IS_B2_A1_A2_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_B2_A1_A2_B2_B1_B1

### Relational analysis result of IS_B2_A1_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -9.9643424, upper bound: 10.0240646
time: 1.69 seconds

## Relational analysis of IS_B2_A1_A2_B2_B1_B2

### Relational analysis result of IS_B2_A1_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -9.9914298, upper bound: 10.0968363
time: 1.75 seconds

## BFS IS instance: IS_B2_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -1.2261815, 1.3062198, -4.9549704, 4.1017799, -5.3279614, 6.2611904
1: -0.9791979, 1.1733789, -4.3657265, 3.7865992, -4.7657971, 5.5391054
2: -0.9199095, 1.6271781, -5.6395326, 3.9404986, -4.8604083, 7.2667108
3: -1.0922583, 1.1746113, -5.9149270, 3.2190866, -4.3113451, 7.0895386
4: -1.1806741, 1.2008911, -5.8374100, 4.1625233, -5.3431973, 7.0383010
5: -1.0925370, 1.1990054, -4.8901324, 3.9499094, -5.0424466, 6.0891380
6: -1.1249063, 1.1679790, -4.5576596, 4.5601029, -5.6850090, 5.7256384
7: -1.1833377, 1.1946260, -4.9786892, 4.9098926, -6.0932302, 6.1733150
8: -1.7498575, 2.5849457, -7.7505288, 3.7638307, -5.5136881, 10.3354740
9: -1.0911226, 1.3417611, -4.3454690, 4.6057243, -5.6968470, 5.6872301

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_B2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_B2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_B2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_B2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_B2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_B2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_B2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_B2_A1_A2_B2_B2_A1

### Relational analysis result of IS_B2_A1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0765472, upper bound: 10.1218231
time: 1.44 seconds

## Relational analysis of IS_B2_A1_A2_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_B2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_B2_A1_A2_B2_B2_A1

### Relational analysis result of IS_B2_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2340818, upper bound: 10.3269938
time: 2.53 seconds

## Relational analysis of IS_B2_A1_A2_B2_B2_A2

### Relational analysis result of IS_B2_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.2340818, upper bound: 10.3269938
time: 1.76 seconds

## BFS IS instance: IS_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -2.2877104, 1.9087899, -0.7839077, 0.9813707, -3.2690811, 2.6926975
1: -1.8476956, 1.8796308, -0.6236843, 0.7426473, -2.5903430, 2.5033150
2: -2.4105849, 2.1123986, -0.7004970, 1.1132182, -3.5238032, 2.8128958
3: -2.5813680, 1.8356632, -0.6336464, 0.7548249, -3.3361928, 2.4693096
4: -2.5926938, 1.9941437, -0.8255264, 0.7804300, -3.3731236, 2.8196702
5: -2.1247373, 1.9644263, -0.7093499, 0.8492496, -2.9739869, 2.6737761
6: -2.1501877, 2.1079900, -0.7357227, 0.8105178, -2.9607055, 2.8437128
7: -2.2694120, 2.1918299, -0.7704170, 0.8206292, -3.0900412, 2.9622469
8: -3.5469000, 2.7682288, -1.0128505, 2.7122550, -6.2591553, 3.7810793
9: -1.9375224, 2.2168264, -0.8165429, 0.9571456, -2.8946681, 3.0333693

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0655433, upper bound: 10.0057429
time: 2.27 seconds

## Relational analysis of IS_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1373336, upper bound: 10.0318994
time: 2.73 seconds

## BFS IS instance: IS_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -2.2877104, 1.9087899, -1.2681804, 1.2783751, -3.5660856, 3.1769702
1: -1.8476956, 1.8796308, -0.9806322, 1.0860441, -2.9337397, 2.8602629
2: -2.4105849, 2.1123986, -1.1637746, 1.4336338, -3.8442187, 3.2761731
3: -2.5813680, 1.8356632, -1.2028930, 0.9989486, -3.5803165, 3.0385561
4: -2.5926938, 1.9941437, -1.3576442, 1.1508827, -3.7435765, 3.3517880
5: -2.1247373, 1.9644263, -1.1365323, 1.1857440, -3.3104813, 3.1009586
6: -2.1501877, 2.1079900, -1.1778419, 1.2055669, -3.3557546, 3.2858319
7: -2.2694120, 2.1918299, -1.2093580, 1.2229133, -3.4923253, 3.4011879
8: -3.5469000, 2.7682288, -1.8562834, 2.7976198, -6.3445196, 4.6245122
9: -1.9375224, 2.2168264, -1.1308748, 1.3397750, -3.2772975, 3.3477011

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=12.009641647338867
rel_dist={8: [-10.85331894689149, 10.853318913028158]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8530988, upper bound: 10.8530969
time: 10.98 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8530979, upper bound: 10.8530979
time: 3.73 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 14.87 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 14.87
Output dim: 8, lower bound: -10.8530988, upper bound: 10.8530969
IS_B2, status: Status.UNKNOWN, split count: 1, time: 14.87
Output dim: 8, lower bound: -10.8530979, upper bound: 10.8530979

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -4.4739342, 3.7029376, -4.2726665, 3.5375986, -8.0115328, 7.9756041
1: -3.8579309, 3.5040054, -3.6662765, 3.3524494, -7.2103806, 7.1702819
2: -5.0883393, 3.6380534, -4.8500576, 3.4913182, -8.5796576, 8.4881115
3: -5.5479379, 3.2747076, -5.2939973, 3.1474588, -8.6953964, 8.5687046
4: -5.2393441, 3.8416514, -5.0029488, 3.6762702, -8.9156141, 8.8446007
5: -4.4075041, 3.6794295, -4.2034626, 3.5188208, -7.9263248, 7.8828921
6: -4.2370515, 4.1352620, -4.0506725, 3.9497395, -8.1867905, 8.1859341
7: -4.5255213, 4.4109535, -4.3245869, 4.2148838, -8.7404051, 8.7355404
8: -6.9224825, 3.6245794, -6.6214066, 3.5089447, -10.4314270, 10.2459860
9: -3.9429712, 4.2205076, -3.7602625, 4.0377922, -7.9807634, 7.9807701

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8500300, upper bound: 10.8495368
time: 4.40 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8521147, upper bound: 10.8521115
time: 3.39 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -4.6222477, 3.8259940, -5.0463686, 4.1626234, -8.7848711, 8.8723602
1: -4.0038152, 3.6204395, -4.3884239, 3.9182816, -7.9220967, 8.0088625
2: -5.2637596, 3.7502680, -5.7543435, 4.0314703, -9.2952290, 9.5046120
3: -5.7378783, 3.3727250, -6.2691126, 3.6060271, -9.3439054, 9.6418381
4: -5.4130883, 3.9648414, -5.9075127, 4.3034272, -9.7165155, 9.8723545
5: -4.5577297, 3.7985079, -4.9838800, 4.1251612, -8.6828909, 8.7823877
6: -4.3756037, 4.2740812, -4.7554636, 4.6543121, -9.0299158, 9.0295439
7: -4.6751213, 4.5558224, -5.1047850, 4.9701166, -9.6452379, 9.6606064
8: -7.1444745, 3.7188740, -7.7569084, 3.9492493, -11.0937233, 11.4757824
9: -4.0795650, 4.3567152, -4.4555368, 4.7338243, -8.8133888, 8.8122520

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8500480, upper bound: 10.8495416
time: 4.12 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8521130, upper bound: 10.8521128
time: 4.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 9.77 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 9.77
Output dim: 8, lower bound: -10.8500300, upper bound: 10.8495368
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 9.77
Output dim: 8, lower bound: -10.8521147, upper bound: 10.8521115
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 9.77
Output dim: 8, lower bound: -10.8500480, upper bound: 10.8495416
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 9.77
Output dim: 8, lower bound: -10.8521130, upper bound: 10.8521128

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -0.6516142, 0.8227109, -1.3851547, 1.2778780, -1.9294922, 2.2078657
1: -0.5444942, 0.6791257, -1.0991892, 1.2074625, -1.7519567, 1.7783148
2: -0.6111455, 0.9994486, -1.2975090, 1.5039041, -2.1150496, 2.2969575
3: -0.4877351, 0.7879708, -1.3635906, 1.2282088, -1.7159439, 2.1515615
4: -0.6958572, 0.6837185, -1.5227106, 1.2586372, -1.9544944, 2.2064290
5: -0.6176764, 0.7560540, -1.2577837, 1.2964598, -1.9141362, 2.0138378
6: -0.6413195, 0.6915185, -1.3158485, 1.2944801, -1.9357996, 2.0073671
7: -0.6615337, 0.6815088, -1.3383936, 1.3331720, -1.9947057, 2.0199025
8: -0.7317882, 2.5303557, -2.0203428, 2.6001689, -3.3319571, 4.5506983
9: -0.7265638, 0.8461591, -1.2085441, 1.4267820, -2.1533458, 2.0547032

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8459828, upper bound: 10.8453789
time: 4.59 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8480272, upper bound: 10.8475647
time: 3.80 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -3.1155267, 2.5666873, -3.2563007, 2.6832075, -5.7987342, 5.8229880
1: -2.5575352, 2.4895847, -2.6929386, 2.5948944, -5.1524296, 5.1825233
2: -3.4352357, 2.6725645, -3.6079676, 2.7687533, -6.2039890, 6.2805319
3: -3.7270939, 2.3902037, -3.9252303, 2.4828219, -6.2099161, 6.3154340
4: -3.5634375, 2.6983809, -3.7427959, 2.8178327, -6.3812704, 6.4411769
5: -2.9894049, 2.5965486, -3.1363218, 2.7079754, -5.6973801, 5.7328701
6: -2.9275854, 2.8723326, -3.0662801, 3.0036607, -5.9312458, 5.9386129
7: -3.1226654, 3.0351148, -3.2704632, 3.1819272, -6.3045926, 6.3055782
8: -4.8510742, 3.0153263, -5.0657005, 3.0566649, -7.9077392, 8.0810270
9: -2.6782391, 2.9614441, -2.8113794, 3.0901544, -5.7683935, 5.7728233

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8495270, upper bound: 10.8500372
time: 3.86 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8495270, upper bound: 10.8521114
time: 4.07 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -0.7097590, 0.8679700, -2.0019174, 1.7113712, -2.4211302, 2.8698874
1: -0.5933210, 0.7272757, -1.6013708, 1.6476047, -2.2409258, 2.3286467
2: -0.6528021, 1.0601923, -2.0358920, 1.9037971, -2.5565991, 3.0960844
3: -0.5543335, 0.8346671, -2.1337461, 1.5703691, -2.1247027, 2.9684134
4: -0.7537465, 0.7379875, -2.2597573, 1.7441204, -2.4978669, 2.9977448
5: -0.6652583, 0.8097053, -1.8361783, 1.7304080, -2.3956661, 2.6458836
6: -0.6985503, 0.7377125, -1.8702841, 1.8387556, -2.5373058, 2.6079965
7: -0.7145321, 0.7398257, -1.9671555, 1.8966157, -2.6111479, 2.7069812
8: -0.8506929, 2.5516868, -3.0686719, 2.7269449, -3.5776377, 5.6203585
9: -0.7618458, 0.8974769, -1.6787620, 1.9450886, -2.7069345, 2.5762389

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_B2_A1_A1

### Relational analysis result of IS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8459990, upper bound: 10.8453866
time: 3.87 seconds

## Relational analysis of IS_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8481017, upper bound: 10.8476056
time: 3.99 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -3.2382431, 2.6724315, -3.9489250, 3.2676210, -6.5058641, 6.6213565
1: -2.6829824, 2.5879865, -3.3628368, 3.1087151, -5.7916975, 5.9508233
2: -3.5857022, 2.7687478, -4.4490905, 3.2634547, -6.8491569, 7.2178383
3: -3.8883061, 2.4698424, -4.8295336, 2.8947499, -6.7830563, 7.2993760
4: -3.7127984, 2.8034244, -4.5920053, 3.3952012, -7.1079998, 7.3954296
5: -3.1161289, 2.6948981, -3.8599205, 3.2518921, -6.3680210, 6.5548186
6: -3.0470028, 2.9892869, -3.7252674, 3.6442459, -6.6912489, 6.7145543
7: -3.2467480, 3.1582675, -3.9819884, 3.8804736, -7.1272216, 7.1402559
8: -5.0441542, 3.0733790, -6.1321244, 3.3639989, -8.4081535, 9.2055035
9: -2.7937706, 3.0752020, -3.4529457, 3.7290130, -6.5227833, 6.5281477

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8495418, upper bound: 10.8500481
time: 4.73 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8495418, upper bound: 10.8521123
time: 4.48 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 10.81 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 10.81
Output dim: 8, lower bound: -10.8459828, upper bound: 10.8453789
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 10.81
Output dim: 8, lower bound: -10.8480272, upper bound: 10.8475647
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 10.81
Output dim: 8, lower bound: -10.8495270, upper bound: 10.8500372
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 10.81
Output dim: 8, lower bound: -10.8495270, upper bound: 10.8521114
IS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 10.81
Output dim: 8, lower bound: -10.8459990, upper bound: 10.8453866
IS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 10.81
Output dim: 8, lower bound: -10.8481017, upper bound: 10.8476056
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 10.81
Output dim: 8, lower bound: -10.8495418, upper bound: 10.8500481
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 10.81
Output dim: 8, lower bound: -10.8495418, upper bound: 10.8521123

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.2050991, 0.2383457, -0.3714563, 0.5710178, -0.7761168, 0.6098019
1: -0.1066593, 0.1534686, -0.3065303, 0.4125119, -0.5191712, 0.4599989
2: -0.3479540, 0.2137559, -0.4498668, 0.5735562, -0.9215102, 0.6636227
3: -0.1196864, 0.1584792, -0.2244284, 0.5020884, -0.6217749, 0.3829076
4: -0.1278429, 0.1955157, -0.4366409, 0.4036437, -0.5314866, 0.6321566
5: -0.1894907, 0.1623752, -0.3652228, 0.4978474, -0.6873381, 0.5275980
6: -0.1844466, 0.1544652, -0.3310412, 0.4581050, -0.6425515, 0.4855064
7: -0.2165271, 0.1498221, -0.4147786, 0.4001184, -0.6166456, 0.5646007
8: 0.5410225, 2.3326459, -0.0866350, 2.4021220, -1.8610995, 2.4192810
9: -0.4576041, 0.2578920, -0.5767903, 0.5516929, -1.0092969, 0.8346823

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 176

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_B1_A1_A1_A1

### Relational analysis result of IS_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8440161, upper bound: 10.8435743
time: 3.07 seconds

## Relational analysis of IS_B1_A1_A1_A2

### Relational analysis result of IS_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8437139, upper bound: 10.8432433
time: 3.43 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.3482443, 0.5376749, -0.6916242, 0.8586459, -1.2068902, 1.2292991
1: -0.2789100, 0.3936100, -0.5839822, 0.7171651, -0.9960750, 0.9775922
2: -0.4613319, 0.6118530, -0.6338032, 1.0254958, -1.4868276, 1.2456563
3: -0.2101499, 0.4658700, -0.5404800, 0.8368461, -1.0469960, 1.0063500
4: -0.4231138, 0.3628573, -0.7407196, 0.7266216, -1.1497355, 1.1035769
5: -0.3390432, 0.4635097, -0.6524426, 0.7979006, -1.1369438, 1.1159524
6: -0.3102885, 0.4188979, -0.6811281, 0.7258384, -1.0361269, 1.1000260
7: -0.3863330, 0.3596223, -0.7035635, 0.7325264, -1.1188594, 1.0631857
8: -0.1011528, 2.4513543, -0.8189834, 2.5133724, -2.6145253, 3.2703376
9: -0.5807495, 0.5382190, -0.7531530, 0.8814771, -1.4622266, 1.2913721

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_B1_A1_A2_A1

### Relational analysis result of IS_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8469374, upper bound: 10.8465537
time: 3.69 seconds

## Relational analysis of IS_B1_A1_A2_A2

### Relational analysis result of IS_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8468920, upper bound: 10.8464427
time: 5.01 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -3.1155267, 2.5666873, -0.5823044, 0.7623762, -3.8779030, 3.1489918
1: -2.5575352, 2.4895847, -0.4856315, 0.6175168, -3.1750519, 2.9752162
2: -3.4352357, 2.6725645, -0.5682570, 0.9213098, -4.3565454, 3.2408214
3: -3.7270939, 2.3902037, -0.4081050, 0.7298506, -4.4569445, 2.7983088
4: -3.5634375, 2.6983809, -0.6276305, 0.6173378, -4.1807752, 3.3260114
5: -2.9894049, 2.5965486, -0.5609183, 0.6906986, -3.6801035, 3.1574669
6: -2.9275854, 2.8723326, -0.5705864, 0.6374269, -3.5650122, 3.4429190
7: -3.1226654, 3.0351148, -0.5943890, 0.6131480, -3.7358134, 3.6295037
8: -4.8510742, 3.0153263, -0.5885465, 2.5068948, -7.3579693, 3.6038728
9: -2.6782391, 2.9614441, -0.6914485, 0.7806575, -3.4588966, 3.6528926

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8455889, upper bound: 10.8458393
time: 2.24 seconds

## Relational analysis of IS_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8475543, upper bound: 10.8480357
time: 2.62 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -3.1155267, 2.5666873, -2.9328561, 2.4127960, -5.5283227, 5.4995432
1: -2.5575352, 2.4895847, -2.3821862, 2.3555315, -4.9130669, 4.8717709
2: -3.4352357, 2.6725645, -3.2099998, 2.5388985, -5.9741344, 5.8825645
3: -3.7270939, 2.3902037, -3.4837213, 2.2754459, -6.0025396, 5.8739252
4: -3.5634375, 2.6983809, -3.3391724, 2.5442429, -6.1076803, 6.0375533
5: -2.9894049, 2.5965486, -2.8007469, 2.4521141, -5.4415188, 5.3972955
6: -2.9275854, 2.8723326, -2.7534752, 2.7012711, -5.6288567, 5.6258078
7: -3.1226654, 3.0351148, -2.9355774, 2.8502209, -5.9728861, 5.9706922
8: -4.8510742, 3.0153263, -4.5668526, 2.9443765, -7.7954507, 7.5821791
9: -2.6782391, 2.9614441, -2.5102952, 2.7934465, -5.4716854, 5.4717393

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_B1_A2_B2_B1

### Relational analysis result of IS_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8489453, upper bound: 10.8515920
time: 4.81 seconds

## Relational analysis of IS_B1_A2_B2_B2

### Relational analysis result of IS_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8488269, upper bound: 10.8515907
time: 3.59 seconds

## BFS IS instance: IS_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.2140643, 0.2602022, -0.5456873, 0.7520826, -0.9661469, 0.8058895
1: -0.1174722, 0.1665233, -0.4607544, 0.5746969, -0.6921691, 0.6272777
2: -0.3533034, 0.2418681, -0.5235958, 0.8069322, -1.1602356, 0.7654638
3: -0.1227972, 0.1744641, -0.3740740, 0.6833806, -0.8061778, 0.5485381
4: -0.1405560, 0.2077042, -0.5885872, 0.5887585, -0.7293144, 0.7962914
5: -0.1968936, 0.1801768, -0.5350827, 0.6513960, -0.8482897, 0.7152595
6: -0.1883174, 0.1713590, -0.5149284, 0.6175456, -0.8058630, 0.6862874
7: -0.2300352, 0.1632586, -0.5681437, 0.5858093, -0.8158445, 0.7314023
8: 0.4961273, 2.3484070, -0.4760341, 2.4800975, -1.9839703, 2.8244412
9: -0.4635872, 0.2786137, -0.6613706, 0.7350767, -1.1986638, 0.9399843

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_B2_A1_A1_A1

### Relational analysis result of IS_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8440347, upper bound: 10.8435776
time: 3.25 seconds

## Relational analysis of IS_B2_A1_A1_A2

### Relational analysis result of IS_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8437179, upper bound: 10.8432453
time: 3.00 seconds

## BFS IS instance: IS_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.3709690, 0.5620617, -1.1831856, 1.1689713, -1.5399404, 1.7452472
1: -0.3021076, 0.4116171, -0.9464169, 1.0652659, -1.3673735, 1.3580339
2: -0.4725067, 0.6502972, -1.0895971, 1.3668609, -1.8393676, 1.7398944
3: -0.2258913, 0.4988205, -1.1229649, 1.1080728, -1.3339641, 1.6217854
4: -0.4394289, 0.3906420, -1.2736385, 1.1111565, -1.5505854, 1.6642804
5: -0.3594005, 0.4925871, -1.0778909, 1.1519448, -1.5113453, 1.5704780
6: -0.3355873, 0.4439169, -1.1315866, 1.1293404, -1.4649277, 1.5755035
7: -0.4041197, 0.3864843, -1.1444113, 1.1542413, -1.5583611, 1.5308957
8: -0.1602615, 2.4707775, -1.6812398, 2.6079116, -2.7681730, 4.1520176
9: -0.5914989, 0.5644029, -1.0651398, 1.2727787, -1.8642776, 1.6295427

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_B2_A1_A2_A1

### Relational analysis result of IS_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8470021, upper bound: 10.8466299
time: 3.23 seconds

## Relational analysis of IS_B2_A1_A2_A2

### Relational analysis result of IS_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8469547, upper bound: 10.8465166
time: 4.38 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -3.2382431, 2.6724315, -0.9803172, 1.0454071, -4.2836504, 3.6527486
1: -2.6829824, 2.5879865, -0.7935669, 0.9196899, -3.6026723, 3.3815534
2: -3.5857022, 2.7687478, -0.8952392, 1.2469547, -4.8326569, 3.6639872
3: -3.8883061, 2.4698424, -0.8810340, 0.9822551, -4.8705611, 3.3508763
4: -3.7127984, 2.8034244, -1.0375305, 0.9559020, -4.6687002, 3.8409548
5: -3.1161289, 2.6948981, -0.8902529, 1.0087825, -4.1249113, 3.5851510
6: -3.0470028, 2.9892869, -0.9496063, 0.9593213, -4.0063243, 3.9388933
7: -3.2467480, 3.1582675, -0.9525447, 0.9855250, -4.2322731, 4.1108122
8: -5.0441542, 3.0733790, -1.3291003, 2.5987730, -7.6429272, 4.4024792
9: -2.7937706, 3.0752020, -0.9272943, 1.1096374, -3.9034081, 4.0024962

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_B2_A2_B1_B1

### Relational analysis result of IS_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8489605, upper bound: 10.8493481
time: 13.96 seconds

## Relational analysis of IS_B2_A2_B1_B2

### Relational analysis result of IS_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8488322, upper bound: 10.8493215
time: 3.47 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -3.2382431, 2.6724315, -3.6151822, 2.9853709, -6.2236137, 6.2876139
1: -2.6829824, 2.5879865, -3.0412400, 2.8579402, -5.5409226, 5.6292267
2: -3.5857022, 2.7687478, -4.0409288, 3.0234888, -6.6091909, 6.8096766
3: -3.8883061, 2.4698424, -4.3813834, 2.6751404, -6.5634465, 6.8512259
4: -3.7127984, 2.8034244, -4.1784372, 3.1129715, -6.8257699, 6.9818616
5: -3.1161289, 2.6948981, -3.5073612, 2.9843984, -6.1005273, 6.2022591
6: -3.0470028, 2.9892869, -3.4007874, 3.3334327, -6.3804355, 6.3900743
7: -3.2467480, 3.1582675, -3.6357284, 3.5418658, -6.7886138, 6.7939959
8: -5.0441542, 3.0733790, -5.6205091, 3.2150056, -8.2591600, 8.6938877
9: -2.7937706, 3.0752020, -3.1396620, 3.4166355, -6.2104063, 6.2148638

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 210

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_B2_A2_B2_B1

### Relational analysis result of IS_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8489605, upper bound: 10.8493481
time: 7.56 seconds

## Relational analysis of IS_B2_A2_B2_B2

### Relational analysis result of IS_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8488322, upper bound: 10.8493215
time: 8.24 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 17.20 seconds
IS_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 17.20
Output dim: 8, lower bound: -10.8440161, upper bound: 10.8435743
IS_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 17.20
Output dim: 8, lower bound: -10.8437139, upper bound: 10.8432433
IS_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 17.20
Output dim: 8, lower bound: -10.8469374, upper bound: 10.8465537
IS_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 17.20
Output dim: 8, lower bound: -10.8468920, upper bound: 10.8464427
IS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 17.20
Output dim: 8, lower bound: -10.8455889, upper bound: 10.8458393
IS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 17.20
Output dim: 8, lower bound: -10.8475543, upper bound: 10.8480357
IS_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 17.20
Output dim: 8, lower bound: -10.8489453, upper bound: 10.8515920
IS_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 17.20
Output dim: 8, lower bound: -10.8488269, upper bound: 10.8515907
IS_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 17.20
Output dim: 8, lower bound: -10.8440347, upper bound: 10.8435776
IS_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 17.20
Output dim: 8, lower bound: -10.8437179, upper bound: 10.8432453
IS_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 17.20
Output dim: 8, lower bound: -10.8470021, upper bound: 10.8466299
IS_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 17.20
Output dim: 8, lower bound: -10.8469547, upper bound: 10.8465166
IS_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 17.20
Output dim: 8, lower bound: -10.8489605, upper bound: 10.8493481
IS_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 17.20
Output dim: 8, lower bound: -10.8488322, upper bound: 10.8493215
IS_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 17.20
Output dim: 8, lower bound: -10.8489605, upper bound: 10.8493481
IS_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 17.20
Output dim: 8, lower bound: -10.8488322, upper bound: 10.8493215

## BFS IS instance: IS_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.1797809, 0.1399598, -0.3018813, 0.4597085, -0.6394894, 0.4418412
1: -0.0763216, 0.0978197, -0.2388149, 0.3102368, -0.3865583, 0.3366346
2: -0.3262541, 0.1463436, -0.3788794, 0.4786066, -0.8048607, 0.5252230
3: -0.1102649, 0.0913040, -0.1833264, 0.3912284, -0.5014933, 0.2746304
4: -0.0891182, 0.1148594, -0.2720484, 0.3264313, -0.3961509, 0.3869078
5: -0.1527683, 0.0966444, -0.3027804, 0.3476795, -0.5004478, 0.3994249
6: -0.1709030, 0.0991212, -0.2673098, 0.3292074, -0.5001104, 0.3664310
7: -0.1663661, 0.1000272, -0.3574363, 0.3135082, -0.4798744, 0.4574634
8: 0.7098049, 2.2653317, 0.1114946, 2.3458183, -1.6360134, 2.1538372
9: -0.4333668, 0.1545884, -0.4772628, 0.4727737, -0.9061405, 0.6318511

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_A1_A1_A1_A1

### Relational analysis result of IS_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6737638, upper bound: 10.7087199
time: 3.21 seconds

## Relational analysis of IS_B1_A1_A1_A1_A2

### Relational analysis result of IS_B1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6634948, upper bound: 10.7091902
time: 3.11 seconds

## BFS IS instance: IS_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.3041008, 0.5524896, -0.2981486, 0.4556496, -0.7597504, 0.8506383
1: -0.2418295, 0.3132194, -0.2349087, 0.3065974, -0.5484269, 0.5481281
2: -0.3694695, 0.4651434, -0.3756994, 0.4732457, -0.8427151, 0.8408428
3: -0.1977779, 0.3567728, -0.1807777, 0.3842481, -0.5820261, 0.5375505
4: -0.2764962, 0.3370765, -0.2675564, 0.3233731, -0.5998693, 0.6046329
5: -0.3083941, 0.3698062, -0.2996666, 0.3431350, -0.6515291, 0.6694728
6: -0.2640788, 0.3616085, -0.2635796, 0.3254817, -0.5895604, 0.6251881
7: -0.3788203, 0.3249829, -0.3547308, 0.3089228, -0.6877431, 0.6797137
8: 0.0851863, 2.3283939, 0.1208199, 2.3383074, -2.2531211, 2.2075741
9: -0.4737600, 0.4972299, -0.4741616, 0.4678428, -0.9416028, 0.9713915

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_B1_A1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6053249, upper bound: 10.6448320
time: 11.26 seconds

## Relational analysis of IS_B1_A1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6303234, upper bound: 10.6804479
time: 3.40 seconds

## BFS IS instance: IS_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.2820500, 0.4062176, -0.5162955, 0.7137823, -0.9958323, 0.9225131
1: -0.2097381, 0.2773573, -0.4413927, 0.5670642, -0.7768023, 0.7187499
2: -0.3866404, 0.5074419, -0.5287896, 0.8435120, -1.2301524, 1.0362315
3: -0.1614714, 0.3492124, -0.3412057, 0.7024530, -0.8639243, 0.6904181
4: -0.2479348, 0.2919179, -0.5676382, 0.5589533, -0.8068882, 0.8595561
5: -0.2721332, 0.3037449, -0.5024333, 0.6445826, -0.9167158, 0.8061782
6: -0.2412804, 0.2937044, -0.5103534, 0.5907413, -0.8320217, 0.8040578
7: -0.3280419, 0.2682068, -0.5363634, 0.5564190, -0.8844609, 0.8045701
8: 0.0999749, 2.3685637, -0.4721790, 2.4380038, -2.3380289, 2.8407426
9: -0.4782152, 0.4435908, -0.6571025, 0.7241902, -1.2024055, 1.1006932

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_B1_A1_A2_A1_A1

### Relational analysis result of IS_B1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8464892, upper bound: 10.8461578
time: 3.29 seconds

## Relational analysis of IS_B1_A1_A2_A1_A2

### Relational analysis result of IS_B1_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8464799, upper bound: 10.8460833
time: 5.24 seconds

## BFS IS instance: IS_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.5908318, 0.8277390, -0.5081062, 0.7062278, -1.2970595, 1.3358452
1: -0.4948949, 0.5619037, -0.4341891, 0.5589499, -1.0538449, 0.9960928
2: -0.5540570, 0.9266119, -0.5218825, 0.8335783, -1.3876354, 1.4484944
3: -0.3446866, 0.7375910, -0.3330695, 0.6937844, -1.0384710, 1.0706605
4: -0.5762242, 0.5498469, -0.5586351, 0.5499089, -1.1261331, 1.1084820
5: -0.5523564, 0.7127768, -0.4928120, 0.6373534, -1.1897099, 1.2055888
6: -0.5029970, 0.6994010, -0.4999885, 0.5838292, -1.0868261, 1.1993896
7: -0.5751351, 0.6247820, -0.5293558, 0.5480291, -1.1231642, 1.1541378
8: -0.5767566, 2.4418116, -0.4545112, 2.4248896, -3.0016460, 2.8963227
9: -0.6763802, 0.7980675, -0.6517353, 0.7149355, -1.3913157, 1.4498028

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_B1_A1_A2_A2_B1

### Relational analysis result of IS_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7327644, upper bound: 10.7567578
time: 2.75 seconds

## Relational analysis of IS_B1_A1_A2_A2_B2

### Relational analysis result of IS_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7674613, upper bound: 10.8052258
time: 3.89 seconds

## BFS IS instance: IS_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.6395401, 0.8121176, -0.2378397, 0.3334071, -0.9729472, 1.0499574
1: -0.5617652, 0.6815366, -0.1588054, 0.2158076, -0.7775728, 0.8403419
2: -0.5775026, 0.9128550, -0.3638611, 0.3317909, -0.9092935, 1.2767161
3: -0.4909189, 0.8586938, -0.1343976, 0.2321032, -0.7230221, 0.9930913
4: -0.6782553, 0.6974025, -0.1878392, 0.2483855, -0.9266407, 0.8852417
5: -0.6181927, 0.7550722, -0.2296086, 0.2411598, -0.8593525, 0.9846808
6: -0.6248454, 0.6848941, -0.2018328, 0.2325734, -0.8574188, 0.8867269
7: -0.6631652, 0.6819721, -0.2791246, 0.2137938, -0.8769591, 0.9610967
8: -0.6954740, 2.4619577, 0.3630791, 2.3565881, -3.0520620, 2.0988786
9: -0.7131107, 0.8302504, -0.4694263, 0.3486111, -1.0617218, 1.2996767

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_B1_A2_B1_A1_B1

### Relational analysis result of IS_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8437410, upper bound: 10.8439352
time: 4.93 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2

### Relational analysis result of IS_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8436140, upper bound: 10.8438454
time: 7.33 seconds

## BFS IS instance: IS_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1.9593816, 1.6687877, -0.3629693, 0.5493901, -2.5087717, 2.0317571
1: -1.5776055, 1.6487811, -0.2929167, 0.4048955, -1.9825009, 1.9416977
2: -1.9837943, 1.8839140, -0.4669214, 0.6300898, -2.6138840, 2.3508353
3: -2.1315737, 1.6381072, -0.2193297, 0.4852763, -2.6168499, 1.8574369
4: -2.2058291, 1.7242517, -0.4340717, 0.3775560, -2.5833852, 2.1583235
5: -1.8029466, 1.7218393, -0.3535992, 0.4793474, -2.2822940, 2.0754385
6: -1.8444693, 1.8129416, -0.3243600, 0.4359875, -2.2804568, 2.1373014
7: -1.9220328, 1.8663206, -0.3981503, 0.3773522, -2.2993851, 2.2644708
8: -3.0070097, 2.6971626, -0.1260886, 2.4463768, -5.4533863, 2.8232512
9: -1.6433008, 1.9310267, -0.5862787, 0.5527267, -2.1960275, 2.5173054

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_A2_B1_A2_A1

### Relational analysis result of IS_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8423840, upper bound: 10.8431543
time: 3.95 seconds

## Relational analysis of IS_B1_A2_B1_A2_A2

### Relational analysis result of IS_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8435647, upper bound: 10.8441722
time: 6.23 seconds

## BFS IS instance: IS_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -2.6728699, 2.2046170, -2.3642178, 1.9606442, -4.6335144, 4.5688348
1: -2.1656063, 2.1702557, -1.9131807, 1.9470960, -4.1127024, 4.0834365
2: -2.8922558, 2.3771908, -2.5082746, 2.1683440, -5.0605998, 4.8854656
3: -3.1394470, 2.1393232, -2.7259469, 1.9498641, -5.0893111, 4.8652701
4: -3.0313573, 2.3251829, -2.6792746, 2.0637350, -5.0950923, 5.0044575
5: -2.5289097, 2.2606068, -2.2033539, 2.0326715, -4.5615811, 4.4639606
6: -2.5112896, 2.4681203, -2.2274861, 2.1860967, -4.6973863, 4.6956062
7: -2.6665530, 2.5855789, -2.3497040, 2.2750111, -4.9415641, 4.9352827
8: -4.1623611, 2.8349259, -3.6666729, 2.7284186, -6.8907795, 6.5015988
9: -2.2772326, 2.5652704, -2.0081916, 2.2922745, -4.5695071, 4.5734620

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_B1_A2_B2_B1_A1

### Relational analysis result of IS_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8517077, upper bound: 10.8515920
time: 4.24 seconds

## Relational analysis of IS_B1_A2_B2_B1_A2

### Relational analysis result of IS_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8517072, upper bound: 10.8515920
time: 5.39 seconds

## BFS IS instance: IS_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -2.6397009, 2.1777270, -4.2868376, 3.5185628, -6.1582637, 6.4645643
1: -2.1382270, 2.1472549, -3.6864691, 3.3196497, -5.4578767, 5.8337240
2: -2.8505607, 2.3548312, -4.8371086, 3.4449830, -6.2955437, 7.1919398
3: -3.0945883, 2.1175025, -5.2385192, 3.0248568, -6.1194448, 7.3560219
4: -2.9928966, 2.2958856, -5.0319586, 3.6413884, -6.6342850, 7.3278441
5: -2.4949899, 2.2355814, -4.1990404, 3.4847734, -5.9797630, 6.4346218
6: -2.4778156, 2.4381394, -3.9746931, 3.9465287, -6.4243441, 6.4128323
7: -2.6319261, 2.5526536, -4.3163862, 4.2472143, -6.8791404, 6.8690395
8: -4.1085253, 2.8113780, -6.6447396, 3.3340766, -7.4426022, 9.4561176
9: -2.2494674, 2.5347962, -3.7565403, 3.9992735, -6.2487411, 6.2913365

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_B1_A2_B2_B2_A1

### Relational analysis result of IS_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8515941, upper bound: 10.8515911
time: 10.41 seconds

## Relational analysis of IS_B1_A2_B2_B2_A2

### Relational analysis result of IS_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8515937, upper bound: 10.8515907
time: 3.82 seconds

## BFS IS instance: IS_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.1852142, 0.1715316, -0.4079811, 0.6147453, -0.7999595, 0.5795127
1: -0.0811515, 0.1175290, -0.3411465, 0.4421390, -0.5229175, 0.4586755
2: -0.3329231, 0.1601785, -0.4653380, 0.6365542, -0.9694774, 0.6255164
3: -0.1121993, 0.1122707, -0.2471727, 0.5555862, -0.6677856, 0.3594434
4: -0.0963469, 0.1601388, -0.4651743, 0.4463591, -0.5427060, 0.6253130
5: -0.1716044, 0.1107521, -0.3980765, 0.5350099, -0.7066143, 0.5088286
6: -0.1750553, 0.1104211, -0.3687703, 0.4982780, -0.6694663, 0.4791914
7: -0.1767684, 0.1144796, -0.4448188, 0.4439467, -0.6207151, 0.5592985
8: 0.6439613, 2.2801733, -0.1913996, 2.4107461, -1.7667848, 2.4715729
9: -0.4383571, 0.2019937, -0.5939071, 0.5970270, -1.0353841, 0.7959008

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A1_A1_A1_A1

### Relational analysis result of IS_B2_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6741349, upper bound: 10.7091340
time: 3.52 seconds

## Relational analysis of IS_B2_A1_A1_A1_A2

### Relational analysis result of IS_B2_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6634655, upper bound: 10.7094469
time: 2.59 seconds

## BFS IS instance: IS_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.3151838, 0.5712578, -0.3999178, 0.6066940, -0.9218777, 0.9711756
1: -0.2519100, 0.3252079, -0.3335429, 0.4351304, -0.6870404, 0.6587508
2: -0.3748830, 0.4883386, -0.4596182, 0.6257886, -1.0006716, 0.9479569
3: -0.2059613, 0.3698679, -0.2420389, 0.5438397, -0.7498010, 0.6119069
4: -0.2875792, 0.3472528, -0.4578232, 0.4372663, -0.7248454, 0.8047476
5: -0.3179725, 0.3836692, -0.3905045, 0.5275686, -0.8455411, 0.7741737
6: -0.2730235, 0.3754531, -0.3590710, 0.4902293, -0.7632528, 0.7345241
7: -0.3897243, 0.3363716, -0.4386162, 0.4350730, -0.8247973, 0.7749878
8: 0.0513533, 2.3399918, -0.1723243, 2.3962226, -2.3448694, 2.5123162
9: -0.4791416, 0.5130572, -0.5881890, 0.5876065, -1.0667481, 1.1012462

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_B2_A1_A1_A2_A1

### Relational analysis result of IS_B2_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6026126, upper bound: 10.6549011
time: 5.78 seconds

## Relational analysis of IS_B2_A1_A1_A2_A2

### Relational analysis result of IS_B2_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6303066, upper bound: 10.6806594
time: 2.34 seconds

## BFS IS instance: IS_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.2955704, 0.4623340, -0.7977638, 0.9344137, -1.2299842, 1.2600979
1: -0.2246630, 0.3416526, -0.6648393, 0.8032213, -1.0278842, 1.0064919
2: -0.4308773, 0.5406256, -0.7164953, 1.1241696, -1.5550469, 1.2571208
3: -0.1706169, 0.3942630, -0.6658233, 0.9090216, -1.0796385, 1.0600863
4: -0.3732578, 0.3064694, -0.8444725, 0.8143836, -1.1876414, 1.1509418
5: -0.2855030, 0.3954538, -0.7329113, 0.8851606, -1.1706636, 1.1283650
6: -0.2550972, 0.3588627, -0.7773839, 0.8188686, -1.0739658, 1.1362466
7: -0.3410946, 0.2869370, -0.7948727, 0.8281947, -1.1692894, 1.0818096
8: 0.0154808, 2.3857839, -1.0210575, 2.5200841, -2.5046034, 3.4068413
9: -0.5519106, 0.4661389, -0.8152303, 0.9710090, -1.5229197, 1.2813691

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_B2_A1_A2_A1_B1

### Relational analysis result of IS_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7390249, upper bound: 10.7598833
time: 4.04 seconds

## Relational analysis of IS_B2_A1_A2_A1_B2

### Relational analysis result of IS_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7757560, upper bound: 10.8086768
time: 3.84 seconds

## BFS IS instance: IS_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.6194829, 0.8817948, -0.7711064, 0.9158980, -1.5353808, 1.6529012
1: -0.5278936, 0.5928710, -0.6437990, 0.7841873, -1.3120809, 1.2366700
2: -0.5748402, 0.9633681, -0.6896132, 1.1054177, -1.6802579, 1.6529813
3: -0.3785365, 0.7971735, -0.6336001, 0.8919185, -1.2704551, 1.4307735
4: -0.6125351, 0.6996436, -0.8166321, 0.7905116, -1.4030466, 1.5162756
5: -0.5723827, 0.7692696, -0.7086575, 0.8651123, -1.4374950, 1.4779272
6: -0.5895480, 0.7246272, -0.7512521, 0.7972507, -1.3867987, 1.4758792
7: -0.6168689, 0.6897179, -0.7715809, 0.8050437, -1.4219126, 1.4612988
8: -0.7059441, 2.4569292, -0.9725085, 2.4984837, -3.2044277, 3.4294376
9: -0.7276784, 0.8195940, -0.7986284, 0.9478595, -1.6755378, 1.6182225

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_B2_A1_A2_A2_B1

### Relational analysis result of IS_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8469547, upper bound: 10.8465166
time: 5.75 seconds

## Relational analysis of IS_B2_A1_A2_A2_B2

### Relational analysis result of IS_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8469547, upper bound: 10.8465166
time: 3.98 seconds

## BFS IS instance: IS_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -2.7847180, 2.2959113, -0.5954242, 0.7762282, -3.5609462, 2.8913355
1: -2.2559996, 2.2570593, -0.4988427, 0.6344004, -2.8903999, 2.7559021
2: -3.0311580, 2.4602664, -0.5709996, 0.9523268, -3.9834847, 3.0312660
3: -3.2883134, 2.2126453, -0.4203804, 0.7446256, -4.0329390, 2.6330256
4: -3.1566241, 2.4211662, -0.6387654, 0.6266150, -3.7832391, 3.0599315
5: -2.6465297, 2.3442883, -0.5664876, 0.7059647, -3.3524945, 2.9107759
6: -2.6188209, 2.5706778, -0.5848994, 0.6507943, -3.2696152, 3.1555772
7: -2.7786031, 2.6964509, -0.6056465, 0.6243962, -3.4029994, 3.3020973
8: -4.3398752, 2.8842971, -0.6233088, 2.4867933, -6.8266687, 3.5076060
9: -2.3777194, 2.6643789, -0.6940199, 0.7960326, -3.1737521, 3.3583987

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_B2_A2_B1_B1_A1

### Relational analysis result of IS_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8437623, upper bound: 10.8439400
time: 3.97 seconds

## Relational analysis of IS_B2_A2_B1_B1_A2

### Relational analysis result of IS_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8466299, upper bound: 10.8470021
time: 4.25 seconds

## BFS IS instance: IS_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -2.7453668, 2.2624171, -2.1141870, 1.7450776, -4.4904442, 4.3766041
1: -2.2235363, 2.2277093, -1.6415365, 1.6928405, -3.9163766, 3.8692458
2: -2.9818697, 2.4320643, -2.0280385, 1.9738045, -4.9556742, 4.4601030
3: -3.2349164, 2.1871293, -2.2500837, 1.5552590, -4.7901754, 4.4372129
4: -3.1115580, 2.3860579, -2.3441515, 1.8152639, -4.9268217, 4.7302094
5: -2.6058688, 2.3141360, -1.9157717, 1.7903459, -4.3962145, 4.2299080
6: -2.5785134, 2.5346851, -1.9950359, 1.9089682, -4.4874816, 4.5297213
7: -2.7375898, 2.6582019, -1.9510288, 2.0025959, -4.7401857, 4.6092310
8: -4.2760658, 2.8518806, -3.2796524, 2.6166828, -6.8927488, 6.1315327
9: -2.3423042, 2.6282001, -1.7121677, 1.9909797, -4.3332839, 4.3403678

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_B2_A2_B1_B2_A1

### Relational analysis result of IS_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8488321, upper bound: 10.8493215
time: 5.14 seconds

## Relational analysis of IS_B2_A2_B1_B2_A2

### Relational analysis result of IS_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8488322, upper bound: 10.8493215
time: 5.01 seconds

## BFS IS instance: IS_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -2.7847180, 2.2959113, -2.9242558, 2.4078376, -5.1925554, 5.2201672
1: -2.2559996, 2.2570593, -2.3747039, 2.3535595, -4.6095591, 4.6317635
2: -3.0311580, 2.4602664, -3.1998692, 2.5499558, -5.5811138, 5.6601353
3: -3.2883134, 2.2126453, -3.4711325, 2.2869020, -5.5752153, 5.6837778
4: -3.1566241, 2.4211662, -3.3249581, 2.5340641, -5.6906881, 5.7461243
5: -2.6465297, 2.3442883, -2.7915137, 2.4469581, -5.0934877, 5.1358023
6: -2.6188209, 2.5706778, -2.7473340, 2.6957097, -5.3145304, 5.3180118
7: -2.7786031, 2.6964509, -2.9227054, 2.8382394, -5.6168423, 5.6191564
8: -4.3398752, 2.8842971, -4.5547976, 2.9199631, -7.2598381, 7.4390945
9: -2.3777194, 2.6643789, -2.5056338, 2.7856600, -5.1633797, 5.1700125

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 210

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 148

## Relational analysis of IS_B2_A2_B2_B1_A1

### Relational analysis result of IS_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8493452, upper bound: 10.8496414
time: 12.26 seconds

## Relational analysis of IS_B2_A2_B2_B1_A2

### Relational analysis result of IS_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8493146, upper bound: 10.8491970
time: 3.66 seconds

## BFS IS instance: IS_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -2.7453668, 2.2624171, -4.8560972, 4.0147400, -6.7601070, 7.1185141
1: -2.2235363, 2.2277093, -4.2333660, 3.7550488, -5.9785852, 6.4610753
2: -2.9818697, 2.4320643, -5.5257359, 3.8803015, -6.8621712, 7.9577999
3: -3.2349164, 2.1871293, -5.9686980, 3.3774171, -6.6123333, 8.1558275
4: -3.1115580, 2.3860579, -5.7229681, 4.1200275, -7.2315855, 8.1090260
5: -2.6058688, 2.3141360, -4.8216124, 3.9487574, -6.5546265, 7.1357484
6: -2.5785134, 2.5346851, -4.5541649, 4.4725814, -7.0510950, 7.0888500
7: -2.7375898, 2.6582019, -4.9158950, 4.8132524, -7.5508423, 7.5740967
8: -4.2760658, 2.8518806, -7.5335579, 3.6575489, -7.9336147, 10.3854389
9: -2.3423042, 2.6282001, -4.2868714, 4.5461102, -6.8884144, 6.9150715

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 210

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 148

## Relational analysis of IS_B2_A2_B2_B2_A1

### Relational analysis result of IS_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8492024, upper bound: 10.8496221
time: 6.57 seconds

## Relational analysis of IS_B2_A2_B2_B2_A2

### Relational analysis result of IS_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8491954, upper bound: 10.8491954
time: 3.48 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 11.51 seconds
IS_B1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 8, lower bound: -10.6737638, upper bound: 10.7087199
IS_B1_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 8, lower bound: -10.6634948, upper bound: 10.7091902
IS_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 8, lower bound: -10.6053249, upper bound: 10.6448320
IS_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 8, lower bound: -10.6303234, upper bound: 10.6804479
IS_B1_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 8, lower bound: -10.8464892, upper bound: 10.8461578
IS_B1_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 8, lower bound: -10.8464799, upper bound: 10.8460833
IS_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 8, lower bound: -10.7327644, upper bound: 10.7567578
IS_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 8, lower bound: -10.7674613, upper bound: 10.8052258
IS_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 8, lower bound: -10.8437410, upper bound: 10.8439352
IS_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 8, lower bound: -10.8436140, upper bound: 10.8438454
IS_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 8, lower bound: -10.8423840, upper bound: 10.8431543
IS_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 8, lower bound: -10.8435647, upper bound: 10.8441722
IS_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 8, lower bound: -10.8517077, upper bound: 10.8515920
IS_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 8, lower bound: -10.8517072, upper bound: 10.8515920
IS_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 8, lower bound: -10.8515941, upper bound: 10.8515911
IS_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 8, lower bound: -10.8515937, upper bound: 10.8515907
IS_B2_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 8, lower bound: -10.6741349, upper bound: 10.7091340
IS_B2_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 8, lower bound: -10.6634655, upper bound: 10.7094469
IS_B2_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 8, lower bound: -10.6026126, upper bound: 10.6549011
IS_B2_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 8, lower bound: -10.6303066, upper bound: 10.6806594
IS_B2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 8, lower bound: -10.7390249, upper bound: 10.7598833
IS_B2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 8, lower bound: -10.7757560, upper bound: 10.8086768
IS_B2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 8, lower bound: -10.8469547, upper bound: 10.8465166
IS_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 8, lower bound: -10.8469547, upper bound: 10.8465166
IS_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 8, lower bound: -10.8437623, upper bound: 10.8439400
IS_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 8, lower bound: -10.8466299, upper bound: 10.8470021
IS_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 8, lower bound: -10.8488321, upper bound: 10.8493215
IS_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 8, lower bound: -10.8488322, upper bound: 10.8493215
IS_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 8, lower bound: -10.8493452, upper bound: 10.8496414
IS_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 8, lower bound: -10.8493146, upper bound: 10.8491970
IS_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 8, lower bound: -10.8492024, upper bound: 10.8496221
IS_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 8, lower bound: -10.8491954, upper bound: 10.8491954

## BFS IS instance: IS_B1_A1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.1662962, 0.0709071, -0.1982493, 0.2520658, -0.4183621, 0.2691564
1: -0.0664658, 0.0616934, -0.1137619, 0.1631172, -0.2295830, 0.1754553
2: -0.3065961, 0.0788353, -0.3367974, 0.2255728, -0.5321689, 0.4156327
3: -0.1043657, 0.0427674, -0.1158923, 0.1729585, -0.2773242, 0.1586597
4: -0.0675050, 0.0825713, -0.1342247, 0.2033670, -0.2365866, 0.2167960
5: -0.1414466, 0.0315551, -0.1919723, 0.1742757, -0.3157222, 0.2235274
6: -0.1595025, 0.0692563, -0.1801948, 0.1664518, -0.3259543, 0.2494511
7: -0.1120413, 0.0798124, -0.2270793, 0.1590758, -0.2711171, 0.3068917
8: 0.8575741, 2.2099271, 0.5221116, 2.2823706, -1.4247965, 1.6878154
9: -0.4162236, 0.0363643, -0.4419671, 0.2712626, -0.6874863, 0.4783314

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_B1_A1_A1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6737638, upper bound: 10.7087199
time: 3.25 seconds

## Relational analysis of IS_B1_A1_A1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6737373, upper bound: 10.7087199
time: 3.91 seconds

## BFS IS instance: IS_B1_A1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.1734337, 0.0926247, -0.1977121, 0.2473494, -0.4207830, 0.2903367
1: -0.0700270, 0.0718631, -0.1103535, 0.1589184, -0.2289454, 0.1822166
2: -0.3201275, 0.1008296, -0.3384809, 0.2199821, -0.5401096, 0.4393105
3: -0.1088427, 0.0511662, -0.1160880, 0.1641426, -0.2729853, 0.1672542
4: -0.0727176, 0.0898318, -0.1311944, 0.1997435, -0.2434911, 0.2210262
5: -0.1476565, 0.0520923, -0.1912750, 0.1689730, -0.3166294, 0.2433673
6: -0.1678898, 0.0760500, -0.1805981, 0.1619529, -0.3298427, 0.2566480
7: -0.1213965, 0.0938401, -0.2231531, 0.1554437, -0.2768402, 0.3169931
8: 0.7918313, 2.2551279, 0.5325287, 2.2885122, -1.4966810, 1.7225993
9: -0.4308417, 0.0710875, -0.4435011, 0.2657575, -0.6965992, 0.5145886

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_B1_A1_A1_A1_A2_A1

### Relational analysis result of IS_B1_A1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6634944, upper bound: 10.7089906
time: 3.40 seconds

## Relational analysis of IS_B1_A1_A1_A1_A2_A2

### Relational analysis result of IS_B1_A1_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6634944, upper bound: 10.7091902
time: 2.97 seconds

## BFS IS instance: IS_B1_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2083456, 0.3679067, -0.1820541, 0.1976087, -0.4059543, 0.5499609
1: -0.1406697, 0.1974930, -0.0868638, 0.1281925, -0.2688622, 0.2843567
2: -0.3374053, 0.2894306, -0.3327585, 0.1999609, -0.5373663, 0.6221892
3: -0.1232736, 0.2164693, -0.1124611, 0.1232183, -0.2464919, 0.3289305
4: -0.1627584, 0.2341529, -0.1025788, 0.1703568, -0.3331152, 0.3367318
5: -0.2123375, 0.2254383, -0.1777069, 0.1262765, -0.3386139, 0.4031452
6: -0.1882548, 0.2175830, -0.1752755, 0.1233098, -0.3115645, 0.3928586
7: -0.2634686, 0.2064077, -0.1897484, 0.1251429, -0.3886116, 0.3961561
8: 0.3478163, 2.2786171, 0.5643580, 2.2667089, -1.9188926, 1.7142591
9: -0.4461763, 0.3464188, -0.4334060, 0.2304776, -0.6766539, 0.7798247

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_A1_A1_A2_B1_A1

### Relational analysis result of IS_B1_A1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3960422, upper bound: 10.4276464
time: 2.92 seconds

## Relational analysis of IS_B1_A1_A1_A2_B1_A2

### Relational analysis result of IS_B1_A1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3855516, upper bound: 10.4279400
time: 2.46 seconds

## BFS IS instance: IS_B1_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2046584, 0.3504776, -0.2072470, 0.2712511, -0.4759095, 0.5577246
1: -0.1327120, 0.1882674, -0.1199350, 0.1728321, -0.3055441, 0.3082024
2: -0.3372618, 0.2759234, -0.3599929, 0.2770358, -0.6142975, 0.6359164
3: -0.1208143, 0.2046548, -0.1242373, 0.1756294, -0.2964437, 0.3288921
4: -0.1542621, 0.2261562, -0.1442392, 0.2100343, -0.3642963, 0.3703954
5: -0.2059758, 0.2131764, -0.2035298, 0.1839678, -0.3899437, 0.4167062
6: -0.1849929, 0.2054343, -0.1915674, 0.1783404, -0.3633333, 0.3970017
7: -0.2538069, 0.1967713, -0.2352125, 0.1683850, -0.4221919, 0.4319838
8: 0.3715397, 2.2823958, 0.4396744, 2.3512416, -1.9797020, 1.8427215
9: -0.4463889, 0.3329045, -0.4636971, 0.2951974, -0.7415863, 0.7966016

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_A1_A1_A2_B2_A1

### Relational analysis result of IS_B1_A1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4248171, upper bound: 10.4688415
time: 2.79 seconds

## Relational analysis of IS_B1_A1_A1_A2_B2_A2

### Relational analysis result of IS_B1_A1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4165903, upper bound: 10.4695753
time: 14.71 seconds

## BFS IS instance: IS_B1_A1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.2498418, 0.3298307, -0.4463745, 0.6293042, -0.8791460, 0.7762052
1: -0.1750890, 0.2373686, -0.3782161, 0.4869877, -0.6620768, 0.6155847
2: -0.3705460, 0.4344129, -0.4954293, 0.7438557, -1.1144016, 0.9298422
3: -0.1383817, 0.2940317, -0.2799195, 0.6346851, -0.7730668, 0.5739512
4: -0.2073886, 0.2595226, -0.4977301, 0.4881657, -0.6955543, 0.7572526
5: -0.2407713, 0.2562863, -0.4308034, 0.5793235, -0.8200948, 0.6870897
6: -0.2115819, 0.2466248, -0.4247212, 0.5221756, -0.7337575, 0.6713459
7: -0.2919789, 0.2244705, -0.4732001, 0.4795369, -0.7715158, 0.6976706
8: 0.2195973, 2.3334951, -0.3134506, 2.4065378, -2.1869404, 2.6469457
9: -0.4607399, 0.3857107, -0.6197723, 0.6439699, -1.1047099, 1.0054829

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_B1_A1_A2_A1_A1_B1

### Relational analysis result of IS_B1_A1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8237207, upper bound: 10.8367150
time: 3.63 seconds

## Relational analysis of IS_B1_A1_A2_A1_A1_B2

### Relational analysis result of IS_B1_A1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8328460, upper bound: 10.8398843
time: 3.35 seconds

## BFS IS instance: IS_B1_A1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -0.3166946, 0.4420551, -0.4459932, 0.6259333, -0.9426279, 0.8880484
1: -0.2511911, 0.3260495, -0.3773726, 0.4859006, -0.7370917, 0.7034222
2: -0.4040417, 0.5648166, -0.4950799, 0.7443320, -1.1483737, 1.0598965
3: -0.1836281, 0.4466202, -0.2791435, 0.6328681, -0.8164961, 0.7257637
4: -0.2929815, 0.3289978, -0.4971530, 0.4870186, -0.7800002, 0.8261508
5: -0.3092361, 0.3474058, -0.4299067, 0.5783342, -0.8875703, 0.7773124
6: -0.2780940, 0.3341396, -0.4233139, 0.5209274, -0.7990214, 0.7574536
7: -0.3619899, 0.3117429, -0.4722343, 0.4783781, -0.8403680, 0.7839773
8: 0.0121955, 2.4118624, -0.3101274, 2.4075401, -2.3953447, 2.7219896
9: -0.4979587, 0.4906293, -0.6188037, 0.6424322, -1.1403910, 1.1094329

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A1_A2_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7800037, upper bound: 10.8057335
time: 3.51 seconds

## Relational analysis of IS_B1_A1_A2_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7998751, upper bound: 10.8295756
time: 3.44 seconds

## BFS IS instance: IS_B1_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3745170, 0.6158980, -0.2486843, 0.3497480, -0.7242650, 0.8645823
1: -0.2940573, 0.4034644, -0.1678661, 0.2322461, -0.5263035, 0.5713305
2: -0.4556105, 0.6492128, -0.3727146, 0.4529836, -0.9085940, 1.0219275
3: -0.2288539, 0.4372826, -0.1400774, 0.2606207, -0.4894746, 0.5773600
4: -0.4335251, 0.3807599, -0.2043290, 0.2551890, -0.6764727, 0.5850888
5: -0.3601315, 0.5022902, -0.2393802, 0.2543286, -0.6144601, 0.7416704
6: -0.3239526, 0.4633484, -0.2125743, 0.2465084, -0.5704610, 0.6759226
7: -0.4117248, 0.3994667, -0.2897089, 0.2253940, -0.6371188, 0.6891755
8: -0.1617689, 2.3751559, 0.1984798, 2.3349528, -2.4967217, 2.1766763
9: -0.5771788, 0.5759244, -0.4616018, 0.3907303, -0.9679091, 1.0375261

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_B1_A1_A2_A2_B1_B1

### Relational analysis result of IS_B1_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5739469, upper bound: 10.5953711
time: 3.54 seconds

## Relational analysis of IS_B1_A1_A2_A2_B1_B2

### Relational analysis result of IS_B1_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5831592, upper bound: 10.6112893
time: 3.17 seconds

## BFS IS instance: IS_B1_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3578766, 0.5987338, -0.2963317, 0.4393269, -0.7972035, 0.8950655
1: -0.2808260, 0.3932443, -0.2197140, 0.2931159, -0.5739418, 0.6129583
2: -0.4514102, 0.6296825, -0.4060213, 0.5458947, -0.9973049, 1.0357038
3: -0.2195952, 0.4261063, -0.1730208, 0.3572914, -0.5768866, 0.5991271
4: -0.4229320, 0.3690007, -0.2620443, 0.3047526, -0.7146214, 0.6310450
5: -0.3458433, 0.4860404, -0.2869340, 0.3205692, -0.6664125, 0.7729744
6: -0.3112929, 0.4445867, -0.2565512, 0.3120375, -0.6233304, 0.7011380
7: -0.3998906, 0.3815116, -0.3420588, 0.2863149, -0.6862055, 0.7235703
8: -0.1369972, 2.3799081, 0.0461599, 2.4227514, -2.5597486, 2.3337481
9: -0.5733023, 0.5601528, -0.4972181, 0.4686607, -1.0419629, 1.0573709

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_B1_A1_A2_A2_B2_B1

### Relational analysis result of IS_B1_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6088649, upper bound: 10.6459371
time: 50.51 seconds

## Relational analysis of IS_B1_A1_A2_A2_B2_B2

### Relational analysis result of IS_B1_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6162377, upper bound: 10.6590863
time: 4.95 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.4850423, 0.6786330, -0.1986913, 0.2412765, -0.7263188, 0.8773243
1: -0.4352700, 0.5400330, -0.1123234, 0.1613833, -0.5966532, 0.6523563
2: -0.4942598, 0.7364336, -0.3406145, 0.2449166, -0.7391764, 1.0770481
3: -0.3135887, 0.7338463, -0.1163496, 0.1650926, -0.4786813, 0.8501959
4: -0.5279671, 0.5461448, -0.1330115, 0.2008357, -0.7288028, 0.6791563
5: -0.4790474, 0.6200247, -0.1925475, 0.1716932, -0.6507406, 0.8125722
6: -0.4695898, 0.5669147, -0.1811453, 0.1630301, -0.6326199, 0.7480600
7: -0.5154416, 0.5246181, -0.2248822, 0.1546259, -0.6700675, 0.7495003
8: -0.3819605, 2.3969402, 0.5073492, 2.2869072, -2.6688676, 1.8895910
9: -0.6291406, 0.6838127, -0.4423716, 0.2713972, -0.9005378, 1.1261843

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6294259, upper bound: 10.6279067
time: 3.45 seconds

## Relational analysis of IS_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6665017, upper bound: 10.6556419
time: 3.43 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.4796806, 0.6735051, -0.3604149, 0.6417296, -1.1214101, 1.0339200
1: -0.4300097, 0.5340080, -0.3006620, 0.3738313, -0.8038410, 0.8346699
2: -0.4904902, 0.7292970, -0.3981724, 0.5729681, -1.0634582, 1.1274694
3: -0.3082992, 0.7268561, -0.2391064, 0.4225969, -0.7308961, 0.9659625
4: -0.5222080, 0.5408773, -0.3362583, 0.3866043, -0.9088123, 0.8771356
5: -0.4736229, 0.6151880, -0.3653363, 0.4385983, -0.9122213, 0.9805243
6: -0.4624773, 0.5623497, -0.3106282, 0.4301791, -0.8926563, 0.8729779
7: -0.5108212, 0.5190442, -0.4357863, 0.3826293, -0.8934506, 0.9548305
8: -0.3704172, 2.3899744, -0.0607960, 2.3528032, -2.7232203, 2.4507704
9: -0.6258676, 0.6765473, -0.4924317, 0.5792683, -1.2051358, 1.1689790

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 103

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6277789, upper bound: 10.6249095
time: 9.03 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6639708, upper bound: 10.6509823
time: 3.70 seconds

## BFS IS instance: IS_B1_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.5274376, 0.7117152, -0.2182558, 0.2704072, -0.7978448, 0.9299710
1: -0.4828617, 0.5988964, -0.1299108, 0.1830491, -0.6659108, 0.7288072
2: -0.5313228, 0.7913035, -0.3600431, 0.2786162, -0.8099390, 1.1513467
3: -0.3628372, 0.8191513, -0.1252228, 0.2114657, -0.5743029, 0.9443741
4: -0.5680465, 0.5968822, -0.1534280, 0.2187740, -0.7868205, 0.7503102
5: -0.5296465, 0.6594065, -0.2069576, 0.1944646, -0.7241111, 0.8663642
6: -0.5374385, 0.5979906, -0.1926295, 0.1858440, -0.7232825, 0.7906201
7: -0.5537963, 0.5648304, -0.2430195, 0.1726436, -0.7264400, 0.8078499
8: -0.4893549, 2.4667368, 0.4288857, 2.3555350, -2.8448899, 2.0378511
9: -0.6651195, 0.7369294, -0.4671508, 0.3003411, -0.9654606, 1.2040802

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_B1_A2_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7308907, upper bound: 10.6920624
time: 22.49 seconds

## Relational analysis of IS_B1_A2_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7382783, upper bound: 10.7032024
time: 3.00 seconds

## BFS IS instance: IS_B1_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -1.1555822, 1.1481380, -0.2389677, 0.3208842, -1.4764664, 1.3871058
1: -0.9509169, 1.0796890, -0.1567286, 0.2156436, -1.1665605, 1.2364177
2: -1.0658457, 1.3401506, -0.3746668, 0.3815425, -1.4473882, 1.7148173
3: -1.0986247, 1.1920280, -0.1345462, 0.2499028, -1.3485274, 1.3265742
4: -1.2377446, 1.1100149, -0.1870281, 0.2445035, -1.4822481, 1.2970430
5: -1.0656947, 1.1462977, -0.2273133, 0.2351398, -1.3008344, 1.3736110
6: -1.1258500, 1.1180482, -0.2055432, 0.2258448, -1.3516948, 1.3235914
7: -1.1225545, 1.1316593, -0.2737506, 0.2066521, -1.3292066, 1.4054098
8: -1.6502841, 2.6009192, 0.2820576, 2.3781281, -4.0284119, 2.3188617
9: -1.0536796, 1.2643647, -0.4758313, 0.3578525, -1.4115320, 1.7401960

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_B1_A2_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8364762, upper bound: 10.8155769
time: 6.17 seconds

## Relational analysis of IS_B1_A2_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8354762, upper bound: 10.8075983
time: 3.02 seconds

## BFS IS instance: IS_B1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -1.8699377, 1.5870124, -1.9153175, 1.6146417, -3.4845793, 3.5023298
1: -1.5031918, 1.6003051, -1.5365574, 1.6286592, -3.1318512, 3.1368625
2: -1.8802232, 1.8523128, -1.9356406, 1.8755299, -3.7557530, 3.7879534
3: -2.0438716, 1.6834962, -2.1020625, 1.6876155, -3.7314873, 3.7855587
4: -2.0831885, 1.6593612, -2.1454511, 1.6902995, -3.7734880, 3.8048124
5: -1.7105341, 1.6707864, -1.7521663, 1.6997023, -3.4102364, 3.4229527
6: -1.7744205, 1.7362506, -1.8117445, 1.7743107, -3.5487313, 3.5479951
7: -1.8211601, 1.7826619, -1.8725295, 1.8288140, -3.6499741, 3.6551914
8: -2.8340135, 2.6065483, -2.9091847, 2.5907612, -5.4247746, 5.5157328
9: -1.5717483, 1.8581191, -1.6100693, 1.8923299, -3.4640782, 3.4681883

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_B1_A2_B2_B1_A1_A1

### Relational analysis result of IS_B1_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8505707, upper bound: 10.8504069
time: 4.81 seconds

## Relational analysis of IS_B1_A2_B2_B1_A1_A2

### Relational analysis result of IS_B1_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8507240, upper bound: 10.8505847
time: 3.48 seconds

## BFS IS instance: IS_B1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -2.4187689, 2.0017407, -2.1823530, 1.8124540, -4.2312231, 4.1840935
1: -1.9597601, 1.9906672, -1.7591982, 1.8171456, -3.7769055, 3.7498655
2: -2.5810978, 2.2102647, -2.2748642, 2.0465469, -4.6276445, 4.4851289
3: -2.8030419, 1.9903942, -2.4741163, 1.8412610, -4.6443028, 4.4645104
4: -2.7406139, 2.1125107, -2.4630985, 1.9100507, -4.6506648, 4.5756092
5: -2.2609124, 2.0754685, -2.0084989, 1.8966517, -4.1575642, 4.0839672
6: -2.2755046, 2.2376809, -2.0575113, 2.0175664, -4.2930708, 4.2951922
7: -2.4074373, 2.3297565, -2.1550851, 2.0903401, -4.4977775, 4.4848413
8: -3.7547331, 2.7455523, -3.3603284, 2.6676652, -6.4223986, 6.1058807
9: -2.0553358, 2.3420434, -1.8430439, 2.1288924, -4.1842279, 4.1850872

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_B1_A2_B2_B1_A2_A1

### Relational analysis result of IS_B1_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8505705, upper bound: 10.8504066
time: 3.48 seconds

## Relational analysis of IS_B1_A2_B2_B1_A2_A2

### Relational analysis result of IS_B1_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8507226, upper bound: 10.8505845
time: 5.54 seconds

## BFS IS instance: IS_B1_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -1.8462605, 1.5691001, -3.7899628, 3.1108494, -4.9571099, 5.3590631
1: -1.4847910, 1.5835559, -3.2128444, 2.9681373, -4.4529285, 4.7964001
2: -1.8484004, 1.8370273, -4.2387557, 3.1068192, -4.9552193, 6.0757828
3: -2.0107944, 1.6674211, -4.5862279, 2.7406821, -4.7514763, 6.2536488
4: -2.0542445, 1.6393582, -4.4210215, 3.2310810, -5.2853255, 6.0603795
5: -1.6882279, 1.6529827, -3.6886392, 3.1018889, -4.7901168, 5.3416219
6: -1.7511344, 1.7150081, -3.5112524, 3.4914825, -5.2426167, 5.2262607
7: -1.7962248, 1.7608047, -3.8107677, 3.7425785, -5.5388031, 5.5715723
8: -2.7919204, 2.5903869, -5.8766232, 3.0786986, -5.8706188, 8.4670105
9: -1.5539072, 1.8363771, -3.3045220, 3.5509710, -5.1048784, 5.1408992

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_B1_A2_B2_B2_A1_A1

### Relational analysis result of IS_B1_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8504704, upper bound: 10.8504051
time: 4.04 seconds

## Relational analysis of IS_B1_A2_B2_B2_A1_A2

### Relational analysis result of IS_B1_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8505924, upper bound: 10.8505832
time: 3.37 seconds

## BFS IS instance: IS_B1_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -2.3739390, 1.9650023, -4.0663829, 3.3371634, -5.7111025, 6.0313854
1: -1.9214176, 1.9587257, -3.4764132, 3.1631274, -5.0845451, 5.4351387
2: -2.5226083, 2.1800535, -4.5709333, 3.2941456, -5.8167539, 6.7509871
3: -2.7402964, 1.9610013, -4.9497476, 2.8980210, -5.6383171, 6.9107490
4: -2.6870618, 2.0730484, -4.7607541, 3.4594154, -6.1464772, 6.8338022
5: -2.2137151, 2.0416794, -3.9725037, 3.3147006, -5.5284157, 6.0141830
6: -2.2317924, 2.1963198, -3.7689869, 3.7443638, -5.9761562, 5.9653068
7: -2.3582342, 2.2847497, -4.0922413, 4.0230246, -6.3812590, 6.3769913
8: -3.6786506, 2.7220106, -6.3029675, 3.2179043, -6.8965549, 9.0249786
9: -2.0165942, 2.3002706, -3.5559301, 3.7999055, -5.8164997, 5.8562007

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_B1_A2_B2_B2_A2_B1

### Relational analysis result of IS_B1_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8512713, upper bound: 10.8511393
time: 4.73 seconds

## Relational analysis of IS_B1_A2_B2_B2_A2_B2

### Relational analysis result of IS_B1_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8511138, upper bound: 10.8511376
time: 6.05 seconds

## BFS IS instance: IS_B2_A1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.1691358, 0.0731485, -0.2379085, 0.3584320, -0.5275677, 0.3110571
1: -0.0672839, 0.0645171, -0.1693889, 0.2278253, -0.2951092, 0.2339060
2: -0.3103572, 0.0823274, -0.3569077, 0.3158197, -0.6261768, 0.4392350
3: -0.1056059, 0.0445906, -0.1354168, 0.2720613, -0.3776672, 0.1800074
4: -0.0691831, 0.0857872, -0.1939271, 0.2591882, -0.2901383, 0.2797143
5: -0.1432731, 0.0343293, -0.2377867, 0.2540613, -0.3973344, 0.2721159
6: -0.1616936, 0.0716414, -0.2044415, 0.2453740, -0.4070676, 0.2760829
7: -0.1139756, 0.0811826, -0.2899475, 0.2250777, -0.3390533, 0.3711301
8: 0.8494536, 2.2224925, 0.3465122, 2.3376839, -1.4882303, 1.8759803
9: -0.4199513, 0.0402957, -0.4666623, 0.3613706, -0.7813218, 0.5069581

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_B2_A1_A1_A1_A1_A1

### Relational analysis result of IS_B2_A1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6741144, upper bound: 10.7090332
time: 2.90 seconds

## Relational analysis of IS_B2_A1_A1_A1_A1_A2

### Relational analysis result of IS_B2_A1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6741144, upper bound: 10.7091340
time: 2.94 seconds

## BFS IS instance: IS_B2_A1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.1772066, 0.1044922, -0.2386997, 0.3576591, -0.5348657, 0.3431919
1: -0.0715625, 0.0797627, -0.1670870, 0.2257737, -0.2973362, 0.2468497
2: -0.3260685, 0.1056702, -0.3606577, 0.3362924, -0.6623609, 0.4663279
3: -0.1104042, 0.0585879, -0.1352840, 0.2641455, -0.3745497, 0.1938718
4: -0.0762744, 0.1196702, -0.1928102, 0.2570983, -0.3333728, 0.3124804
5: -0.1591420, 0.0556992, -0.2356851, 0.2523646, -0.4115066, 0.2913843
6: -0.1708217, 0.0789761, -0.2051123, 0.2432185, -0.4140401, 0.2840884
7: -0.1237560, 0.0998226, -0.2876921, 0.2230047, -0.3467607, 0.3875147
8: 0.7444128, 2.2694819, 0.3198655, 2.3459747, -1.6015619, 1.9496164
9: -0.4357897, 0.1086758, -0.4688756, 0.3650406, -0.8008302, 0.5775514

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_B2_A1_A1_A1_A2_A1

### Relational analysis result of IS_B2_A1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6634645, upper bound: 10.7092738
time: 3.01 seconds

## Relational analysis of IS_B2_A1_A1_A1_A2_A2

### Relational analysis result of IS_B2_A1_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6634645, upper bound: 10.7094469
time: 3.65 seconds

## BFS IS instance: IS_B2_A1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.1972218, 0.3294666, -0.2351831, 0.3526484, -0.5498703, 0.5646497
1: -0.1200039, 0.1734523, -0.1598969, 0.2204560, -0.3404598, 0.3333492
2: -0.3339270, 0.2559362, -0.3595915, 0.3631143, -0.6970413, 0.6155277
3: -0.1171436, 0.1861624, -0.1350630, 0.2352717, -0.3524153, 0.3212254
4: -0.1402000, 0.2129627, -0.1897117, 0.2515540, -0.3917540, 0.4026744
5: -0.1965551, 0.1944852, -0.2318137, 0.2463198, -0.4428749, 0.4262989
6: -0.1803211, 0.1866602, -0.2037597, 0.2387060, -0.4190271, 0.3904199
7: -0.2386020, 0.1829780, -0.2830129, 0.2195650, -0.4581670, 0.4659910
8: 0.3995937, 2.2756445, 0.3036009, 2.3313861, -1.9317924, 1.9720436
9: -0.4431903, 0.3144106, -0.4612197, 0.3659157, -0.8091060, 0.7756302

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A1_A1_A2_A1_A1

### Relational analysis result of IS_B2_A1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3939630, upper bound: 10.4378957
time: 3.09 seconds

## Relational analysis of IS_B2_A1_A1_A2_A1_A2

### Relational analysis result of IS_B2_A1_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3815860, upper bound: 10.4373331
time: 3.17 seconds

## BFS IS instance: IS_B2_A1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.2199187, 0.3869842, -0.2342658, 0.3444447, -0.5643634, 0.6212500
1: -0.1481921, 0.2070234, -0.1576271, 0.2173042, -0.3654963, 0.3646505
2: -0.3587314, 0.3116893, -0.3608107, 0.3574207, -0.7161522, 0.6725000
3: -0.1307259, 0.2265124, -0.1338929, 0.2317081, -0.3624340, 0.3604053
4: -0.1726614, 0.2424761, -0.1868103, 0.2489269, -0.4215882, 0.4292863
5: -0.2216506, 0.2364801, -0.2296464, 0.2425639, -0.4642146, 0.4661265
6: -0.1983953, 0.2296670, -0.2026828, 0.2343220, -0.4327173, 0.4323497
7: -0.2723095, 0.2166156, -0.2798296, 0.2157094, -0.4880189, 0.4964453
8: 0.3073877, 2.3488798, 0.3157406, 2.3361673, -2.0287797, 2.0331392
9: -0.4697686, 0.3617309, -0.4626258, 0.3600956, -0.8298642, 0.8243567

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A1_A1_A2_A2_A1

### Relational analysis result of IS_B2_A1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4248982, upper bound: 10.4690326
time: 2.84 seconds

## Relational analysis of IS_B2_A1_A1_A2_A2_A2

### Relational analysis result of IS_B2_A1_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4165852, upper bound: 10.4697600
time: 2.91 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 7.22 seconds
IS_B1_A1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.6737638, upper bound: 10.7087199
IS_B1_A1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.6737373, upper bound: 10.7087199
IS_B1_A1_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.6634944, upper bound: 10.7089906
IS_B1_A1_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.6634944, upper bound: 10.7091902
IS_B1_A1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.3960422, upper bound: 10.4276464
IS_B1_A1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.3855516, upper bound: 10.4279400
IS_B1_A1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.4248171, upper bound: 10.4688415
IS_B1_A1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.4165903, upper bound: 10.4695753
IS_B1_A1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.8237207, upper bound: 10.8367150
IS_B1_A1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.8328460, upper bound: 10.8398843
IS_B1_A1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.7800037, upper bound: 10.8057335
IS_B1_A1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.7998751, upper bound: 10.8295756
IS_B1_A1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.5739469, upper bound: 10.5953711
IS_B1_A1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.5831592, upper bound: 10.6112893
IS_B1_A1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.6088649, upper bound: 10.6459371
IS_B1_A1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.6162377, upper bound: 10.6590863
IS_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.6294259, upper bound: 10.6279067
IS_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.6665017, upper bound: 10.6556419
IS_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.6277789, upper bound: 10.6249095
IS_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.6639708, upper bound: 10.6509823
IS_B1_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.7308907, upper bound: 10.6920624
IS_B1_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.7382783, upper bound: 10.7032024
IS_B1_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.8364762, upper bound: 10.8155769
IS_B1_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.8354762, upper bound: 10.8075983
IS_B1_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.8505707, upper bound: 10.8504069
IS_B1_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.8507240, upper bound: 10.8505847
IS_B1_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.8505705, upper bound: 10.8504066
IS_B1_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.8507226, upper bound: 10.8505845
IS_B1_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.8504704, upper bound: 10.8504051
IS_B1_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.8505924, upper bound: 10.8505832
IS_B1_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.8512713, upper bound: 10.8511393
IS_B1_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.8511138, upper bound: 10.8511376
IS_B2_A1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.6741144, upper bound: 10.7090332
IS_B2_A1_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.6741144, upper bound: 10.7091340
IS_B2_A1_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.6634645, upper bound: 10.7092738
IS_B2_A1_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.6634645, upper bound: 10.7094469
IS_B2_A1_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.3939630, upper bound: 10.4378957
IS_B2_A1_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.3815860, upper bound: 10.4373331
IS_B2_A1_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.4248982, upper bound: 10.4690326
IS_B2_A1_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 7.22
Output dim: 8, lower bound: -10.4165852, upper bound: 10.4697600
IS_B2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.7390249, upper bound: 10.7598833
IS_B2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.7757560, upper bound: 10.8086768
IS_B2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.8469547, upper bound: 10.8465166
IS_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.8469547, upper bound: 10.8465166
IS_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.8437623, upper bound: 10.8439400
IS_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.8466299, upper bound: 10.8470021
IS_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.8488321, upper bound: 10.8493215
IS_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.8488322, upper bound: 10.8493215
IS_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.8493452, upper bound: 10.8496414
IS_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.8493146, upper bound: 10.8491970
IS_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.8492024, upper bound: 10.8496221
IS_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.8491954, upper bound: 10.8491954
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=12.009641647338867
rel_dist={8: [-10.853314892972527, 10.853314942404314]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1805.86 seconds
