## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 326.172941817
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-176.8887177, 140.5452881, -176.8887177, 140.5452881, -317.4339905, 317.4339905)
1: (-148.7599487, 125.1486740, -148.7599487, 125.1486740, -273.9085693, 273.9085693)
2: (-195.1577606, 127.6752167, -195.1577606, 127.6752167, -322.8329773, 322.8329773)
3: (-207.4779510, 109.6864548, -207.4779510, 109.6864548, -317.1643982, 317.1643982)
4: (-189.6262207, 145.8749542, -189.6262207, 145.8749542, -335.5011597, 335.5011597)
5: (-170.1939697, 132.8175659, -170.1939697, 132.8175659, -303.0114746, 303.0114746)
6: (-163.2100983, 156.9458160, -163.2100983, 156.9458160, -320.1559143, 320.1559143)
7: (-178.3847504, 149.9835510, -178.3847504, 149.9835510, -328.3682861, 328.3682861)
8: (-213.8840027, 145.3365479, -213.8840027, 145.3365479, -359.2205505, 359.2205505)
9: (-161.8587646, 159.9163361, -161.8587646, 159.9163361, -321.7750854, 321.7750854)

## BASE Result
execution time: IAR + LP analysis = 1.29 + 11.02 = 12.31 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -326.2561776, upper bound: 326.2561776


# Binary Search by BASE starts (time budget: 2687.69 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=328.3682861328125
rel_dist={7: [-326.25613672106726, 326.2561367077651]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=328.3682861328125
rel_dist={7: [-326.2560128858547, 326.2560128858547]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=328.3682861328125
rel_dist={7: [-326.25584232239004, 326.2558422835341]}

## Binary Search Result
Binary search time: 45.10 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 2642.59 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2484598, upper bound: 326.2481118
time: 8.65 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2524724, upper bound: 326.2524724
time: 8.24 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 17.07 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 17.07
Output dim: 7, lower bound: -326.2484598, upper bound: 326.2481118
IS_A2, status: Status.UNKNOWN, split count: 1, time: 17.07
Output dim: 7, lower bound: -326.2524724, upper bound: 326.2524724

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -142.9952087, 113.6855774, -173.7214508, 138.0343323, -281.0295105, 287.4070129
1: -120.2532425, 101.1967163, -146.0944672, 122.9092484, -243.1624603, 247.2911835
2: -157.7607422, 103.4563980, -191.6611786, 125.4096756, -283.1704102, 295.1175232
3: -167.6863556, 88.7308502, -203.7593231, 107.7279587, -275.4143066, 292.4901123
4: -153.1332703, 117.9750061, -186.2147217, 143.2657471, -296.3990173, 304.1897278
5: -137.5261078, 107.3854294, -167.1402588, 130.4388428, -267.9649658, 274.5256958
6: -132.0471954, 126.9280167, -160.2979279, 154.1398163, -286.1870117, 287.2259521
7: -144.2999115, 121.4841537, -175.1976471, 147.3182220, -291.6181030, 296.6817627
8: -173.0928497, 117.4631882, -210.0702057, 142.7315216, -315.8243713, 327.5333862
9: -131.0026245, 129.3968506, -158.9744415, 157.0635529, -288.0660706, 288.3712769

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2461612, upper bound: 326.2461612
time: 9.00 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2461612, upper bound: 326.2481118
time: 7.04 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -162.4753876, 129.1040802, -176.8887177, 140.5452881, -303.0206299, 305.9927979
1: -136.6650543, 114.9896698, -148.7599487, 125.1486740, -261.8137207, 263.7495728
2: -179.2824554, 117.4086914, -195.1577606, 127.6752167, -306.9576721, 312.5664062
3: -190.5786285, 100.7902908, -207.4779510, 109.6864548, -300.2650757, 308.2682190
4: -174.1093292, 134.0215607, -189.6262207, 145.8749542, -319.9842834, 323.6477661
5: -156.3003693, 122.0177612, -170.1939697, 132.8175659, -289.1179199, 292.2117004
6: -149.9510803, 144.2054901, -163.2100983, 156.9458160, -306.8969116, 307.4155884
7: -163.9395142, 137.9003448, -178.3847504, 149.9835510, -313.9230652, 316.2850952
8: -196.5453186, 133.4469147, -213.8840027, 145.3365479, -341.8818665, 347.3309021
9: -148.7507019, 146.9695892, -161.8587646, 159.9163361, -308.6670227, 308.8283691

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2481118, upper bound: 326.2484598
time: 8.58 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2481118, upper bound: 326.2524724
time: 8.48 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 18.43 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 18.43
Output dim: 7, lower bound: -326.2461612, upper bound: 326.2461612
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 18.43
Output dim: 7, lower bound: -326.2461612, upper bound: 326.2481118
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 18.43
Output dim: 7, lower bound: -326.2481118, upper bound: 326.2484598
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 18.43
Output dim: 7, lower bound: -326.2481118, upper bound: 326.2524724

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -142.9952087, 113.6855774, -142.9952087, 113.6855774, -256.6807861, 256.6807861
1: -120.2532425, 101.1967163, -120.2532425, 101.1967163, -221.4499359, 221.4499359
2: -157.7607422, 103.4563980, -157.7607422, 103.4563980, -261.2171326, 261.2171326
3: -167.6863556, 88.7308502, -167.6863556, 88.7308502, -256.4172058, 256.4172058
4: -153.1332703, 117.9750061, -153.1332703, 117.9750061, -271.1082764, 271.1082764
5: -137.5261078, 107.3854294, -137.5261078, 107.3854294, -244.9115295, 244.9115295
6: -132.0471954, 126.9280167, -132.0471954, 126.9280167, -258.9752197, 258.9752197
7: -144.2999115, 121.4841537, -144.2999115, 121.4841537, -265.7840576, 265.7840576
8: -173.0928497, 117.4631882, -173.0928497, 117.4631882, -290.5560303, 290.5560303
9: -131.0026245, 129.3968506, -131.0026245, 129.3968506, -260.3994751, 260.3994751

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2351056, upper bound: 326.2364410
time: 9.76 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2321364, upper bound: 326.2321366
time: 8.48 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -142.9952087, 113.6855774, -162.4753876, 129.1040802, -272.0993042, 276.1609497
1: -120.2532425, 101.1967163, -136.6650543, 114.9896698, -235.2428589, 237.8617706
2: -157.7607422, 103.4563980, -179.2824554, 117.4086914, -275.1694031, 282.7388611
3: -167.6863556, 88.7308502, -190.5786285, 100.7902908, -268.4766235, 279.3094177
4: -153.1332703, 117.9750061, -174.1093292, 134.0215607, -287.1548462, 292.0843506
5: -137.5261078, 107.3854294, -156.3003693, 122.0177612, -259.5438232, 263.6857910
6: -132.0471954, 126.9280167, -149.9510803, 144.2054901, -276.2526855, 276.8790894
7: -144.2999115, 121.4841537, -163.9395142, 137.9003448, -282.2002563, 285.4236450
8: -173.0928497, 117.4631882, -196.5453186, 133.4469147, -306.5397339, 314.0085144
9: -131.0026245, 129.3968506, -148.7507019, 146.9695892, -277.9721680, 278.1475525

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2351056, upper bound: 326.2413443
time: 11.05 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2321364, upper bound: 326.2360458
time: 7.72 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -162.4753876, 129.1040802, -142.9952087, 113.6855774, -276.1609497, 272.0993042
1: -136.6650543, 114.9896698, -120.2532425, 101.1967163, -237.8617706, 235.2428589
2: -179.2824554, 117.4086914, -157.7607422, 103.4563980, -282.7388611, 275.1694031
3: -190.5786285, 100.7902908, -167.6863556, 88.7308502, -279.3094177, 268.4766235
4: -174.1093292, 134.0215607, -153.1332703, 117.9750061, -292.0843506, 287.1548462
5: -156.3003693, 122.0177612, -137.5261078, 107.3854294, -263.6857910, 259.5438232
6: -149.9510803, 144.2054901, -132.0471954, 126.9280167, -276.8790894, 276.2526855
7: -163.9395142, 137.9003448, -144.2999115, 121.4841537, -285.4236450, 282.2002563
8: -196.5453186, 133.4469147, -173.0928497, 117.4631882, -314.0085144, 306.5397644
9: -148.7507019, 146.9695892, -131.0026245, 129.3968506, -278.1475525, 277.9721680

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2373464, upper bound: 326.2389371
time: 9.61 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2360458, upper bound: 326.2373435
time: 9.27 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -162.4753876, 129.1040802, -162.4753876, 129.1040802, -291.5794678, 291.5794678
1: -136.6650543, 114.9896698, -136.6650543, 114.9896698, -251.6546936, 251.6546936
2: -179.2824554, 117.4086914, -179.2824554, 117.4086914, -296.6911621, 296.6911621
3: -190.5786285, 100.7902908, -190.5786285, 100.7902908, -291.3688354, 291.3688354
4: -174.1093292, 134.0215607, -174.1093292, 134.0215607, -308.1308899, 308.1308899
5: -156.3003693, 122.0177612, -156.3003693, 122.0177612, -278.3181152, 278.3181152
6: -149.9510803, 144.2054901, -149.9510803, 144.2054901, -294.1565552, 294.1565552
7: -163.9395142, 137.9003448, -163.9395142, 137.9003448, -301.8398438, 301.8398438
8: -196.5453186, 133.4469147, -196.5453186, 133.4469147, -329.9922485, 329.9922180
9: -148.7507019, 146.9695892, -148.7507019, 146.9695892, -295.7202759, 295.7202759

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2373464, upper bound: 326.2464391
time: 9.07 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2360458, upper bound: 326.2443285
time: 9.46 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 20.00 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 20.00
Output dim: 7, lower bound: -326.2351056, upper bound: 326.2364410
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 20.00
Output dim: 7, lower bound: -326.2321364, upper bound: 326.2321366
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 20.00
Output dim: 7, lower bound: -326.2351056, upper bound: 326.2413443
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 20.00
Output dim: 7, lower bound: -326.2321364, upper bound: 326.2360458
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 20.00
Output dim: 7, lower bound: -326.2373464, upper bound: 326.2389371
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 20.00
Output dim: 7, lower bound: -326.2360458, upper bound: 326.2373435
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 20.00
Output dim: 7, lower bound: -326.2373464, upper bound: 326.2464391
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 20.00
Output dim: 7, lower bound: -326.2360458, upper bound: 326.2443285

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -134.4191589, 106.8813019, -142.9952087, 113.6855774, -248.1047363, 249.8765106
1: -113.0178909, 95.1411362, -120.2532425, 101.1967163, -214.2145996, 215.3943787
2: -148.2897797, 97.3019028, -157.7607422, 103.4563980, -251.7461548, 255.0626526
3: -157.6537628, 83.3846893, -167.6863556, 88.7308502, -246.3846130, 251.0710297
4: -143.9373474, 110.9074631, -153.1332703, 117.9750061, -261.9123535, 264.0407410
5: -129.2950134, 100.9468842, -137.5261078, 107.3854294, -236.6804504, 238.4729919
6: -124.1608963, 119.3289795, -132.0471954, 126.9280167, -251.0888977, 251.3761597
7: -135.6821747, 114.2907257, -144.2999115, 121.4841537, -257.1663208, 258.5906372
8: -162.7476654, 110.3271866, -173.0928497, 117.4631882, -280.2108459, 283.4200134
9: -123.2038651, 121.6495743, -131.0026245, 129.3968506, -252.6007080, 252.6521912

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2321366, upper bound: 326.2321364
time: 7.64 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2321366, upper bound: 326.2321364
time: 7.96 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -133.3593292, 106.0192261, -140.3241730, 111.5666656, -244.9259796, 246.3433838
1: -112.0901566, 94.3571320, -118.0041809, 99.3143311, -211.4044800, 212.3613129
2: -147.0768280, 96.4584503, -154.8117981, 101.5459518, -248.6227722, 251.2702332
3: -156.4061432, 82.5972748, -164.5602417, 87.0683441, -243.4744568, 247.1575165
4: -142.7857513, 110.0052490, -150.2667542, 115.7772598, -258.5630188, 260.2720032
5: -128.2919312, 100.0992661, -134.9631653, 105.3832092, -233.6751404, 235.0624390
6: -123.1929474, 118.3621597, -129.5919647, 124.5621643, -247.7551117, 247.9541321
7: -134.5887604, 113.4013748, -141.6186371, 119.2485580, -253.8373108, 255.0200043
8: -161.3923187, 109.2901688, -169.8694611, 115.2432861, -276.6355591, 279.1596375
9: -122.2503586, 120.6452026, -128.5761414, 126.9879608, -249.2382965, 249.2213440

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2268408, upper bound: 326.2271544
time: 7.95 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2274139, upper bound: 326.2274140
time: 7.17 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -134.4191589, 106.8813019, -162.4753876, 129.1040802, -263.5232544, 269.3566895
1: -113.0178909, 95.1411362, -136.6650543, 114.9896698, -228.0075378, 231.8061829
2: -148.2897797, 97.3019028, -179.2824554, 117.4086914, -265.6984863, 276.5843506
3: -157.6537628, 83.3846893, -190.5786285, 100.7902908, -258.4440613, 273.9632568
4: -143.9373474, 110.9074631, -174.1093292, 134.0215607, -277.9589233, 285.0167847
5: -129.2950134, 100.9468842, -156.3003693, 122.0177612, -251.3127747, 257.2472534
6: -124.1608963, 119.3289795, -149.9510803, 144.2054901, -268.3663940, 269.2800598
7: -135.6821747, 114.2907257, -163.9395142, 137.9003448, -273.5825195, 278.2302246
8: -162.7476654, 110.3271866, -196.5453186, 133.4469147, -296.1945496, 306.8724976
9: -123.2038651, 121.6495743, -148.7507019, 146.9695892, -270.1734314, 270.4002380

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2373433, upper bound: 326.2360458
time: 9.71 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2373433, upper bound: 326.2360459
time: 8.65 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -133.3593292, 106.0192261, -160.0121002, 127.1498108, -260.5091553, 266.0312500
1: -112.0901566, 94.3571320, -134.5919495, 113.2547150, -225.3448639, 228.9490814
2: -147.0768280, 96.4584503, -176.5640717, 115.6471634, -262.7239685, 273.0225220
3: -156.4061432, 82.5972748, -187.6958313, 99.2581940, -255.6643066, 270.2930908
4: -142.7857513, 110.0052490, -171.4662781, 131.9950562, -274.7808228, 281.4714661
5: -128.2919312, 100.0992661, -153.9372406, 120.1723022, -248.4642334, 254.0364990
6: -123.1929474, 118.3621597, -147.6866913, 142.0246277, -265.2175293, 266.0487976
7: -134.5887604, 113.4013748, -161.4687500, 135.8385162, -270.4272766, 274.8701172
8: -161.3923187, 109.2901688, -193.5732574, 131.3993835, -292.7916870, 302.8634033
9: -122.2503586, 120.6452026, -146.5131683, 144.7487335, -266.9990845, 267.1583252

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2328613, upper bound: 326.2316755
time: 6.90 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2332883, upper bound: 326.2317535
time: 10.92 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -153.2971191, 121.8210449, -142.9952087, 113.6855774, -266.9826660, 264.8162231
1: -128.9177856, 108.5078888, -120.2532425, 101.1967163, -230.1144714, 228.7611389
2: -169.1404724, 110.8150024, -157.7607422, 103.4563980, -272.5968628, 268.5757141
3: -179.8427277, 95.0695114, -167.6863556, 88.7308502, -268.5735474, 262.7558594
4: -164.2681580, 126.4567795, -153.1332703, 117.9750061, -282.2431641, 279.5900574
5: -147.4913177, 115.1224213, -137.5261078, 107.3854294, -254.8767395, 252.6485291
6: -141.5100708, 136.0706329, -132.0471954, 126.9280167, -268.4380798, 268.1177979
7: -154.7082214, 130.1954651, -144.2999115, 121.4841537, -276.1923828, 274.4953613
8: -185.4722748, 125.8125992, -173.0928497, 117.4631882, -302.9354553, 298.9054565
9: -140.4009247, 138.6768799, -131.0026245, 129.3968506, -269.7977905, 269.6794434

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2360458, upper bound: 326.2373433
time: 8.22 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2360458, upper bound: 326.2373435
time: 7.88 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -154.0926056, 122.4333878, -140.3241730, 111.5666656, -265.6592102, 262.7574768
1: -129.5583344, 109.0396042, -118.0041809, 99.3143311, -228.8726654, 227.0437927
2: -169.9832764, 111.3114319, -154.8117981, 101.5459518, -271.5291138, 266.1232300
3: -180.7612000, 95.4519272, -164.5602417, 87.0683441, -267.8295288, 260.0121765
4: -165.1088867, 127.0810013, -150.2667542, 115.7772598, -280.8861389, 277.3477478
5: -148.2696075, 115.6757889, -134.9631653, 105.3832092, -253.6528168, 250.6389465
6: -142.2503662, 136.7492981, -129.5919647, 124.5621643, -266.8125305, 266.3412476
7: -155.4892731, 130.8679657, -141.6186371, 119.2485580, -274.7377930, 272.4865723
8: -186.3566742, 126.3182373, -169.8694611, 115.2432861, -301.5999451, 296.1876831
9: -141.1363831, 139.3524628, -128.5761414, 126.9879608, -268.1243286, 267.9285889

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2313600, upper bound: 326.2330217
time: 8.88 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2317535, upper bound: 326.2332883
time: 10.74 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -153.2971191, 121.8210449, -162.4753876, 129.1040802, -282.4011841, 284.2964478
1: -128.9177856, 108.5078888, -136.6650543, 114.9896698, -243.9074097, 245.1729431
2: -169.1404724, 110.8150024, -179.2824554, 117.4086914, -286.5491333, 290.0974731
3: -179.8427277, 95.0695114, -190.5786285, 100.7902908, -280.6329651, 285.6481323
4: -164.2681580, 126.4567795, -174.1093292, 134.0215607, -298.2897034, 300.5661011
5: -147.4913177, 115.1224213, -156.3003693, 122.0177612, -269.5090942, 271.4227905
6: -141.5100708, 136.0706329, -149.9510803, 144.2054901, -285.7155457, 286.0217285
7: -154.7082214, 130.1954651, -163.9395142, 137.9003448, -292.6085815, 294.1349792
8: -185.4722748, 125.8125992, -196.5453186, 133.4469147, -318.9191895, 322.3579102
9: -140.4009247, 138.6768799, -148.7507019, 146.9695892, -287.3705139, 287.4275513

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2443302, upper bound: 326.2443285
time: 7.12 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2443302, upper bound: 326.2443285
time: 7.80 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -154.0926056, 122.4333878, -160.0121002, 127.1498108, -281.2423706, 282.4454346
1: -129.5583344, 109.0396042, -134.5919495, 113.2547150, -242.8130341, 243.6315613
2: -169.9832764, 111.3114319, -176.5640717, 115.6471634, -285.6303406, 287.8754883
3: -180.7612000, 95.4519272, -187.6958313, 99.2581940, -280.0193787, 283.1477661
4: -165.1088867, 127.0810013, -171.4662781, 131.9950562, -297.1039429, 298.5472717
5: -148.2696075, 115.6757889, -153.9372406, 120.1723022, -268.4418945, 269.6130066
6: -142.2503662, 136.7492981, -147.6866913, 142.0246277, -284.2749634, 284.4359741
7: -155.4892731, 130.8679657, -161.4687500, 135.8385162, -291.3277893, 292.3366699
8: -186.3566742, 126.3182373, -193.5732574, 131.3993835, -317.7560425, 319.8914490
9: -141.1363831, 139.3524628, -146.5131683, 144.7487335, -285.8851318, 285.8656311

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2407392, upper bound: 326.2409498
time: 8.91 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2410210, upper bound: 326.2410206
time: 7.47 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 17.76 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.76
Output dim: 7, lower bound: -326.2321366, upper bound: 326.2321364
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.76
Output dim: 7, lower bound: -326.2321366, upper bound: 326.2321364
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.76
Output dim: 7, lower bound: -326.2268408, upper bound: 326.2271544
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.76
Output dim: 7, lower bound: -326.2274139, upper bound: 326.2274140
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.76
Output dim: 7, lower bound: -326.2373433, upper bound: 326.2360458
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.76
Output dim: 7, lower bound: -326.2373433, upper bound: 326.2360459
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.76
Output dim: 7, lower bound: -326.2328613, upper bound: 326.2316755
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.76
Output dim: 7, lower bound: -326.2332883, upper bound: 326.2317535
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.76
Output dim: 7, lower bound: -326.2360458, upper bound: 326.2373433
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.76
Output dim: 7, lower bound: -326.2360458, upper bound: 326.2373435
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.76
Output dim: 7, lower bound: -326.2313600, upper bound: 326.2330217
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.76
Output dim: 7, lower bound: -326.2317535, upper bound: 326.2332883
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.76
Output dim: 7, lower bound: -326.2443302, upper bound: 326.2443285
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.76
Output dim: 7, lower bound: -326.2443302, upper bound: 326.2443285
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.76
Output dim: 7, lower bound: -326.2407392, upper bound: 326.2409498
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.76
Output dim: 7, lower bound: -326.2410210, upper bound: 326.2410206

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -134.4191589, 106.8813019, -134.4191589, 106.8813019, -241.3004608, 241.3004608
1: -113.0178909, 95.1411362, -113.0178909, 95.1411362, -208.1590271, 208.1590271
2: -148.2897797, 97.3019028, -148.2897797, 97.3019028, -245.5916748, 245.5916748
3: -157.6537628, 83.3846893, -157.6537628, 83.3846893, -241.0384369, 241.0384369
4: -143.9373474, 110.9074631, -143.9373474, 110.9074631, -254.8447876, 254.8447876
5: -129.2950134, 100.9468842, -129.2950134, 100.9468842, -230.2418976, 230.2418976
6: -124.1608963, 119.3289795, -124.1608963, 119.3289795, -243.4898682, 243.4898529
7: -135.6821747, 114.2907257, -135.6821747, 114.2907257, -249.9728851, 249.9728851
8: -162.7476654, 110.3271866, -162.7476654, 110.3271866, -273.0747986, 273.0747986
9: -123.2038651, 121.6495743, -123.2038651, 121.6495743, -244.8534393, 244.8534393

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2303271, upper bound: 326.2315475
time: 9.04 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2303533, upper bound: 326.2317355
time: 8.45 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -134.4191589, 106.8813019, -133.3593292, 106.0192261, -240.4383698, 240.2406311
1: -113.0178909, 95.1411362, -112.0901566, 94.3571320, -207.3750305, 207.2312927
2: -148.2897797, 97.3019028, -147.0768280, 96.4584503, -244.7481995, 244.3787231
3: -157.6537628, 83.3846893, -156.4061432, 82.5972748, -240.2510376, 239.7908173
4: -143.9373474, 110.9074631, -142.7857513, 110.0052490, -253.9425964, 253.6931763
5: -129.2950134, 100.9468842, -128.2919312, 100.0992661, -229.3942871, 229.2388153
6: -124.1608963, 119.3289795, -123.1929474, 118.3621597, -242.5230560, 242.5219269
7: -135.6821747, 114.2907257, -134.5887604, 113.4013748, -249.0835571, 248.8794708
8: -162.7476654, 110.3271866, -161.3923187, 109.2901688, -272.0378418, 271.7194519
9: -123.2038651, 121.6495743, -122.2503586, 120.6452026, -243.8490601, 243.8999329

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2303271, upper bound: 326.2315475
time: 7.62 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2303533, upper bound: 326.2317355
time: 7.61 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -132.6238251, 105.4364929, -122.0566559, 97.1069260, -229.7307434, 227.4931335
1: -111.4668350, 93.8348160, -102.5596237, 86.3702927, -197.8371277, 196.3944397
2: -146.2626343, 95.9290390, -134.6198120, 88.4511337, -234.7137604, 230.5488281
3: -155.5396118, 82.1385498, -143.0591431, 75.7258759, -231.2654877, 225.1976929
4: -141.9960022, 109.3956604, -130.6442413, 100.6783905, -242.6743927, 240.0399017
5: -127.5814819, 99.5510864, -117.3493423, 91.7789383, -219.3604126, 216.9004211
6: -122.5154953, 117.7093887, -112.7665253, 108.3579712, -230.8734741, 230.4759216
7: -133.8477020, 112.7808914, -123.2409744, 103.8640594, -237.7117310, 236.0218658
8: -160.5040588, 108.6778793, -147.8128662, 100.1055222, -260.6095581, 256.4907227
9: -121.5785446, 119.9802399, -111.9072418, 110.4998245, -232.0783691, 231.8874817

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2267518, upper bound: 326.2267518
time: 8.09 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2267518, upper bound: 326.2271544
time: 8.28 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -132.7878571, 105.5666199, -127.4588089, 101.3807602, -234.1686096, 233.0254211
1: -111.6088715, 93.9534454, -107.1785889, 90.2349625, -201.8438110, 201.1320343
2: -146.4459229, 96.0504990, -140.6142273, 92.3704910, -238.8163910, 236.6647339
3: -155.7356262, 82.2426300, -149.4754028, 79.0891418, -234.8247681, 231.7180176
4: -142.1707916, 109.5329132, -136.4225311, 105.1473389, -247.3181152, 245.9554443
5: -127.7406464, 99.6736755, -122.5578842, 95.7983246, -223.5389404, 222.2315674
6: -122.6667328, 117.8556595, -117.7427139, 113.1679459, -235.8346558, 235.5983734
7: -134.0151367, 112.9223099, -128.7086792, 108.4705048, -242.4856415, 241.6309662
8: -160.7026520, 108.8146744, -154.3475647, 104.5447464, -265.2474060, 263.1622314
9: -121.7298279, 120.1289673, -116.8623962, 115.3751068, -237.1049042, 236.9913635

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2271544, upper bound: 326.2268408
time: 8.23 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2271544, upper bound: 326.2274139
time: 8.75 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -134.4191589, 106.8813019, -153.2971191, 121.8210449, -256.2401428, 260.1783752
1: -113.0178909, 95.1411362, -128.9177856, 108.5078888, -221.5257874, 224.0589294
2: -148.2897797, 97.3019028, -169.1404724, 110.8150024, -259.1047974, 266.4423828
3: -157.6537628, 83.3846893, -179.8427277, 95.0695114, -252.7232513, 263.2273560
4: -143.9373474, 110.9074631, -164.2681580, 126.4567795, -270.3941345, 275.1755981
5: -129.2950134, 100.9468842, -147.4913177, 115.1224213, -244.4174347, 248.4382019
6: -124.1608963, 119.3289795, -141.5100708, 136.0706329, -260.2315369, 260.8390503
7: -135.6821747, 114.2907257, -154.7082214, 130.1954651, -265.8776245, 268.9989624
8: -162.7476654, 110.3271866, -185.4722748, 125.8125992, -288.5602112, 295.7994690
9: -123.2038651, 121.6495743, -140.4009247, 138.6768799, -261.8807068, 262.0505066

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2370819, upper bound: 326.2368751
time: 9.42 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2370559, upper bound: 326.2370118
time: 10.76 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -134.4191589, 106.8813019, -154.0926056, 122.4333878, -256.8525085, 260.9738159
1: -113.0178909, 95.1411362, -129.5583344, 109.0396042, -222.0574951, 224.6994629
2: -148.2897797, 97.3019028, -169.9832764, 111.3114319, -259.6011963, 267.2851562
3: -157.6537628, 83.3846893, -180.7612000, 95.4519272, -253.1056824, 264.1458130
4: -143.9373474, 110.9074631, -165.1088867, 127.0810013, -271.0183411, 276.0163574
5: -129.2950134, 100.9468842, -148.2696075, 115.6757889, -244.9707947, 249.2164917
6: -124.1608963, 119.3289795, -142.2503662, 136.7492981, -260.9101868, 261.5793152
7: -135.6821747, 114.2907257, -155.4892731, 130.8679657, -266.5501404, 269.7799988
8: -162.7476654, 110.3271866, -186.3566742, 126.3182373, -289.0658264, 296.6838379
9: -123.2038651, 121.6495743, -141.1363831, 139.3524628, -262.5563354, 262.7859497

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2370819, upper bound: 326.2368751
time: 7.29 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2370559, upper bound: 326.2370118
time: 8.70 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -132.6238251, 105.4364929, -141.5821075, 112.5618744, -245.1856842, 247.0185852
1: -111.4668350, 93.8348160, -119.0090027, 100.1986084, -211.6654358, 212.8438110
2: -146.2626343, 95.9290390, -156.1905212, 102.4414215, -248.7040558, 252.1195526
3: -155.5396118, 82.1385498, -166.0173492, 87.8200684, -243.3596649, 248.1558990
4: -141.9960022, 109.3956604, -151.6725464, 116.7645569, -258.7605591, 261.0682068
5: -127.5814819, 99.5510864, -136.1704559, 106.4467163, -234.0281677, 235.7215424
6: -122.5154953, 117.7093887, -130.7156372, 125.6765671, -248.1920624, 248.4250183
7: -133.8477020, 112.7808914, -142.9293823, 120.3221359, -254.1698151, 255.7102661
8: -160.5040588, 108.6778793, -171.3200226, 116.1260300, -276.6300964, 279.9978943
9: -121.5785446, 119.9802399, -129.7017822, 128.1124573, -249.6910095, 249.6820221

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2326997, upper bound: 326.2313550
time: 8.91 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2326997, upper bound: 326.2316755
time: 8.35 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -132.7878571, 105.5666199, -147.1567535, 116.9683075, -249.7561646, 252.7233734
1: -111.6088715, 93.9534454, -123.7688446, 104.1832962, -215.7921600, 217.7222900
2: -146.4459229, 96.0504990, -162.3782349, 106.4735870, -252.9195099, 258.4287109
3: -155.7356262, 82.2426300, -172.6163788, 91.2874374, -247.0230560, 254.8589935
4: -142.1707916, 109.5329132, -157.6338806, 121.3756638, -263.5464478, 267.1667786
5: -127.7406464, 99.6736755, -141.5392914, 110.5892334, -238.3298340, 241.2129669
6: -122.6667328, 117.8556595, -135.8461761, 130.6366882, -253.3033905, 253.7018433
7: -134.0151367, 112.9223099, -148.5646362, 125.0630035, -259.0780640, 261.4869385
8: -160.7026520, 108.8146744, -178.0624084, 120.7149506, -281.4176025, 286.8770142
9: -121.7298279, 120.1289673, -134.8068695, 133.1468811, -254.8766937, 254.9358368

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2330217, upper bound: 326.2313600
time: 8.37 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2330217, upper bound: 326.2317535
time: 10.41 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -153.2971191, 121.8210449, -134.4191589, 106.8813019, -260.1783752, 256.2401428
1: -128.9177856, 108.5078888, -113.0178909, 95.1411362, -224.0589294, 221.5257874
2: -169.1404724, 110.8150024, -148.2897797, 97.3019028, -266.4423828, 259.1047974
3: -179.8427277, 95.0695114, -157.6537628, 83.3846893, -263.2273560, 252.7232513
4: -164.2681580, 126.4567795, -143.9373474, 110.9074631, -275.1755981, 270.3941345
5: -147.4913177, 115.1224213, -129.2950134, 100.9468842, -248.4382019, 244.4174347
6: -141.5100708, 136.0706329, -124.1608963, 119.3289795, -260.8390503, 260.2315369
7: -154.7082214, 130.1954651, -135.6821747, 114.2907257, -268.9989624, 265.8776245
8: -185.4722748, 125.8125992, -162.7476654, 110.3271866, -295.7994690, 288.5602112
9: -140.4009247, 138.6768799, -123.2038651, 121.6495743, -262.0505066, 261.8807068

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2329731, upper bound: 326.2344445
time: 9.80 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2329643, upper bound: 326.2346096
time: 9.62 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -153.2971191, 121.8210449, -133.3593292, 106.0192261, -259.3162537, 255.1803741
1: -128.9177856, 108.5078888, -112.0901566, 94.3571320, -223.2749176, 220.5980530
2: -169.1404724, 110.8150024, -147.0768280, 96.4584503, -265.5989380, 257.8918457
3: -179.8427277, 95.0695114, -156.4061432, 82.5972748, -262.4400024, 251.4756317
4: -164.2681580, 126.4567795, -142.7857513, 110.0052490, -274.2734070, 269.2425232
5: -147.4913177, 115.1224213, -128.2919312, 100.0992661, -247.5905762, 243.4143524
6: -141.5100708, 136.0706329, -123.1929474, 118.3621597, -259.8721924, 259.2635193
7: -154.7082214, 130.1954651, -134.5887604, 113.4013748, -268.1095886, 264.7842407
8: -185.4722748, 125.8125992, -161.3923187, 109.2901688, -294.7624512, 287.2048950
9: -140.4009247, 138.6768799, -122.2503586, 120.6452026, -261.0461121, 260.9272461

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2329731, upper bound: 326.2344445
time: 10.07 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2329643, upper bound: 326.2346096
time: 8.33 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -153.3479156, 121.8433380, -122.0566559, 97.1069260, -250.4548340, 243.8999939
1: -128.9274597, 108.5108261, -102.5596237, 86.3702927, -215.2977600, 211.0704498
2: -169.1588898, 110.7758331, -134.6198120, 88.4511337, -257.6100159, 245.3956299
3: -179.8846130, 94.9875717, -143.0591431, 75.7258759, -255.6104889, 238.0467072
4: -164.3091431, 126.4641724, -130.6442413, 100.6783905, -264.9875488, 257.1083984
5: -147.5506897, 115.1208267, -117.3493423, 91.7789383, -239.3296204, 232.4701691
6: -141.5645905, 136.0884247, -112.7665253, 108.3579712, -249.9225616, 248.8549347
7: -154.7391510, 130.2402649, -123.2409744, 103.8640594, -258.6031799, 253.4812317
8: -185.4573517, 125.6981735, -147.8128662, 100.1055222, -285.5628662, 273.5110168
9: -140.4565887, 138.6790466, -111.9072418, 110.4998245, -250.9564056, 250.5862732

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2313550, upper bound: 326.2326997
time: 10.49 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2313550, upper bound: 326.2330217
time: 9.59 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -153.5234070, 121.9828110, -127.4588089, 101.3807602, -254.9041748, 249.4416199
1: -129.0788422, 108.6376343, -107.1785889, 90.2349625, -219.3137970, 215.8162231
2: -169.3550415, 110.9049530, -140.6142273, 92.3704910, -261.7255249, 251.5191650
3: -180.0928345, 95.0988464, -149.4754028, 79.0891418, -259.1819153, 244.5742493
4: -164.4967346, 126.6105423, -136.4225311, 105.1473389, -269.6440125, 263.0330811
5: -147.7203979, 115.2519226, -122.5578842, 95.7983246, -243.5187225, 237.8097992
6: -141.7260590, 136.2450104, -117.7427139, 113.1679459, -254.8940125, 253.9877319
7: -154.9177094, 130.3904114, -128.7086792, 108.4705048, -263.3882141, 259.0989990
8: -185.6698303, 125.8451157, -154.3475647, 104.5447464, -290.2145691, 280.1926880
9: -140.6178131, 138.8386688, -116.8623962, 115.3751068, -255.9929047, 255.7010651

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2316755, upper bound: 326.2328613
time: 8.52 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2316755, upper bound: 326.2332883
time: 8.19 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -153.2971191, 121.8210449, -153.2971191, 121.8210449, -275.1181335, 275.1181335
1: -128.9177856, 108.5078888, -128.9177856, 108.5078888, -237.4256744, 237.4256744
2: -169.1404724, 110.8150024, -169.1404724, 110.8150024, -279.9554443, 279.9554443
3: -179.8427277, 95.0695114, -179.8427277, 95.0695114, -274.9122314, 274.9122314
4: -164.2681580, 126.4567795, -164.2681580, 126.4567795, -290.7249451, 290.7249451
5: -147.4913177, 115.1224213, -147.4913177, 115.1224213, -262.6137085, 262.6137085
6: -141.5100708, 136.0706329, -141.5100708, 136.0706329, -277.5806580, 277.5806580
7: -154.7082214, 130.1954651, -154.7082214, 130.1954651, -284.9036865, 284.9036865
8: -185.4722748, 125.8125992, -185.4722748, 125.8125992, -311.2848816, 311.2848816
9: -140.4009247, 138.6768799, -140.4009247, 138.6768799, -279.0778198, 279.0778198

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2426612, upper bound: 326.2429694
time: 8.11 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2426363, upper bound: 326.2430013
time: 9.83 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -153.2971191, 121.8210449, -154.0926056, 122.4333878, -275.7304688, 275.9135742
1: -128.9177856, 108.5078888, -129.5583344, 109.0396042, -237.9573975, 238.0662231
2: -169.1404724, 110.8150024, -169.9832764, 111.3114319, -280.4519043, 280.7982178
3: -179.8427277, 95.0695114, -180.7612000, 95.4519272, -275.2946472, 275.8307190
4: -164.2681580, 126.4567795, -165.1088867, 127.0810013, -291.3491516, 291.5656738
5: -147.4913177, 115.1224213, -148.2696075, 115.6757889, -263.1671143, 263.3920288
6: -141.5100708, 136.0706329, -142.2503662, 136.7492981, -278.2593689, 278.3209534
7: -154.7082214, 130.1954651, -155.4892731, 130.8679657, -285.5761719, 285.6847229
8: -185.4722748, 125.8125992, -186.3566742, 126.3182373, -311.7905273, 312.1692810
9: -140.4009247, 138.6768799, -141.1363831, 139.3524628, -279.7533875, 279.8132629

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2426612, upper bound: 326.2429694
time: 8.19 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2426363, upper bound: 326.2430013
time: 10.60 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -153.3479156, 121.8433380, -141.5821075, 112.5618744, -265.9097290, 263.4253845
1: -128.9274597, 108.5108261, -119.0090027, 100.1986084, -229.1260681, 227.5198364
2: -169.1588898, 110.7758331, -156.1905212, 102.4414215, -271.6003113, 266.9663391
3: -179.8846130, 94.9875717, -166.0173492, 87.8200684, -267.7046814, 261.0048828
4: -164.3091431, 126.4641724, -151.6725464, 116.7645569, -281.0737000, 278.1367188
5: -147.5506897, 115.1208267, -136.1704559, 106.4467163, -253.9973755, 251.2912903
6: -141.5645905, 136.0884247, -130.7156372, 125.6765671, -267.2411499, 266.8040771
7: -154.7391510, 130.2402649, -142.9293823, 120.3221359, -275.0612793, 273.1696472
8: -185.4573517, 125.6981735, -171.3200226, 116.1260300, -301.5833740, 297.0181885
9: -140.4565887, 138.6790466, -129.7017822, 128.1124573, -268.5690308, 268.3808289

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2407282, upper bound: 326.2407288
time: 10.02 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2407282, upper bound: 326.2409498
time: 8.89 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -153.5234070, 121.9828110, -147.1567535, 116.9683075, -270.4916992, 269.1395569
1: -129.0788422, 108.6376343, -123.7688446, 104.1832962, -233.2621307, 232.4064789
2: -169.3550415, 110.9049530, -162.3782349, 106.4735870, -275.8286133, 273.2831421
3: -180.0928345, 95.0988464, -172.6163788, 91.2874374, -271.3801880, 267.7152100
4: -164.4967346, 126.6105423, -157.6338806, 121.3756638, -285.8723450, 284.2444153
5: -147.7203979, 115.2519226, -141.5392914, 110.5892334, -258.3096313, 256.7911987
6: -141.7260590, 136.2450104, -135.8461761, 130.6366882, -272.3627319, 272.0911865
7: -154.9177094, 130.3904114, -148.5646362, 125.0630035, -279.9807129, 278.9550476
8: -185.6698303, 125.8451157, -178.0624084, 120.7149506, -306.3847656, 303.9075012
9: -140.6178131, 138.8386688, -134.8068695, 133.1468811, -273.7646790, 273.6455078

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2409556, upper bound: 326.2407361
time: 7.75 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2409556, upper bound: 326.2410206
time: 8.58 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 18.12 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.12
Output dim: 7, lower bound: -326.2303271, upper bound: 326.2315475
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.12
Output dim: 7, lower bound: -326.2303533, upper bound: 326.2317355
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.12
Output dim: 7, lower bound: -326.2303271, upper bound: 326.2315475
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.12
Output dim: 7, lower bound: -326.2303533, upper bound: 326.2317355
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.12
Output dim: 7, lower bound: -326.2267518, upper bound: 326.2267518
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.12
Output dim: 7, lower bound: -326.2267518, upper bound: 326.2271544
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.12
Output dim: 7, lower bound: -326.2271544, upper bound: 326.2268408
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.12
Output dim: 7, lower bound: -326.2271544, upper bound: 326.2274139
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.12
Output dim: 7, lower bound: -326.2370819, upper bound: 326.2368751
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.12
Output dim: 7, lower bound: -326.2370559, upper bound: 326.2370118
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.12
Output dim: 7, lower bound: -326.2370819, upper bound: 326.2368751
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.12
Output dim: 7, lower bound: -326.2370559, upper bound: 326.2370118
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.12
Output dim: 7, lower bound: -326.2326997, upper bound: 326.2313550
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.12
Output dim: 7, lower bound: -326.2326997, upper bound: 326.2316755
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.12
Output dim: 7, lower bound: -326.2330217, upper bound: 326.2313600
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.12
Output dim: 7, lower bound: -326.2330217, upper bound: 326.2317535
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.12
Output dim: 7, lower bound: -326.2329731, upper bound: 326.2344445
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.12
Output dim: 7, lower bound: -326.2329643, upper bound: 326.2346096
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.12
Output dim: 7, lower bound: -326.2329731, upper bound: 326.2344445
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.12
Output dim: 7, lower bound: -326.2329643, upper bound: 326.2346096
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.12
Output dim: 7, lower bound: -326.2313550, upper bound: 326.2326997
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.12
Output dim: 7, lower bound: -326.2313550, upper bound: 326.2330217
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.12
Output dim: 7, lower bound: -326.2316755, upper bound: 326.2328613
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.12
Output dim: 7, lower bound: -326.2316755, upper bound: 326.2332883
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.12
Output dim: 7, lower bound: -326.2426612, upper bound: 326.2429694
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.12
Output dim: 7, lower bound: -326.2426363, upper bound: 326.2430013
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.12
Output dim: 7, lower bound: -326.2426612, upper bound: 326.2429694
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.12
Output dim: 7, lower bound: -326.2426363, upper bound: 326.2430013
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.12
Output dim: 7, lower bound: -326.2407282, upper bound: 326.2407288
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.12
Output dim: 7, lower bound: -326.2407282, upper bound: 326.2409498
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.12
Output dim: 7, lower bound: -326.2409556, upper bound: 326.2407361
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.12
Output dim: 7, lower bound: -326.2409556, upper bound: 326.2410206

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -116.3450089, 92.5739212, -133.6971893, 106.3090668, -222.6540680, 226.2711029
1: -97.7368240, 82.3347397, -112.4060822, 94.6282654, -192.3650818, 194.7408142
2: -128.3081055, 84.3475113, -147.4904175, 96.7822266, -225.0903320, 231.8379211
3: -136.3774872, 72.1634598, -156.8031921, 82.9345093, -219.3119965, 228.9666443
4: -124.5202179, 95.9699860, -143.1619263, 110.3092422, -234.8294678, 239.1319122
5: -111.8669052, 87.4888458, -128.5975800, 100.4087372, -212.2756348, 216.0863953
6: -107.5139542, 103.2944107, -123.4959106, 118.6880188, -226.2019653, 226.7902985
7: -117.4972458, 99.0698547, -134.9546204, 113.6815338, -231.1787720, 234.0244598
8: -140.9226379, 95.3514862, -161.8758240, 109.7262802, -250.6488953, 257.2272949
9: -106.7119217, 105.3365097, -122.5444489, 120.9965591, -227.7084808, 227.8809357

Time for backsubstitution: 1.60 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=328.3682861328125
rel_dist={7: [-326.25613672106726, 326.2561367077651]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2475865, upper bound: 326.2473505
time: 9.96 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2524306, upper bound: 326.2524306
time: 8.09 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 18.24 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 18.24
Output dim: 7, lower bound: -326.2475865, upper bound: 326.2473505
IS_A2, status: Status.UNKNOWN, split count: 1, time: 18.24
Output dim: 7, lower bound: -326.2524306, upper bound: 326.2524306

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -142.9952087, 113.6855774, -163.4873199, 129.9208832, -272.9160767, 277.1728516
1: -120.2532425, 101.1967163, -137.4826965, 115.6734085, -235.9266052, 238.6794128
2: -157.7607422, 103.4563980, -180.3638153, 118.0897064, -275.8504333, 283.8202209
3: -167.6863556, 88.7308502, -191.7428284, 101.3975830, -269.0839233, 280.4736328
4: -153.1332703, 117.9750061, -175.1933136, 134.8370056, -287.9702454, 293.1683350
5: -137.5261078, 107.3854294, -157.2736816, 122.7544785, -260.2805481, 264.6590271
6: -132.0471954, 126.9280167, -150.8877106, 145.0740356, -277.1212158, 277.8157349
7: -144.2999115, 121.4841537, -164.8998108, 138.7078400, -283.0077515, 286.3839722
8: -173.0928497, 117.4631882, -197.7485504, 134.3133240, -307.4061890, 315.2117310
9: -131.0026245, 129.3968506, -149.6553497, 147.8449249, -278.8474426, 279.0521851

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2364642, upper bound: 326.2351320
time: 10.69 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2356538, upper bound: 326.2345203
time: 10.76 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -162.4753876, 129.1040802, -171.9773560, 136.6489105, -299.1242676, 301.0814209
1: -136.6650543, 114.9896698, -144.6387177, 121.6862717, -258.3513184, 259.6283875
2: -179.2824554, 117.4086914, -189.7478027, 124.1760559, -303.4584961, 307.1564636
3: -190.5786285, 100.7902908, -201.7194061, 106.6540833, -297.2327271, 302.5096741
4: -174.1093292, 134.0215607, -184.3390503, 141.8367462, -315.9460449, 318.3605957
5: -156.3003693, 122.0177612, -165.4610443, 129.1367340, -285.4371033, 287.4787903
6: -149.9510803, 144.2054901, -158.6924591, 152.6037598, -302.5548401, 302.8979187
7: -163.9395142, 137.9003448, -173.4600830, 145.8660126, -309.8055420, 311.3604126
8: -196.5453186, 133.4469147, -207.9763031, 141.2872925, -337.8326111, 341.4231873
9: -148.7507019, 146.9695892, -157.3923950, 155.5047760, -304.2554626, 304.3619385

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2455385, upper bound: 326.2452017
time: 11.10 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2442590, upper bound: 326.2442590
time: 8.83 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 21.62 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 21.62
Output dim: 7, lower bound: -326.2364642, upper bound: 326.2351320
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 21.62
Output dim: 7, lower bound: -326.2356538, upper bound: 326.2345203
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 21.62
Output dim: 7, lower bound: -326.2455385, upper bound: 326.2452017
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 21.62
Output dim: 7, lower bound: -326.2442590, upper bound: 326.2442590

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -142.2971039, 113.1316223, -154.6595306, 122.9161835, -265.2132874, 267.7911377
1: -119.6641388, 100.7035828, -130.0321350, 109.4406738, -229.1047821, 230.7357178
2: -156.9896851, 102.9552383, -170.6132965, 111.7523041, -268.7419128, 273.5685120
3: -166.8697052, 88.2956085, -181.4148102, 95.8967438, -262.7664490, 269.7103882
4: -152.3847504, 117.3995819, -165.7276459, 127.5617676, -279.9465332, 283.1271973
5: -136.8560638, 106.8612289, -148.8011322, 116.1249161, -252.9809875, 255.6623535
6: -131.4051514, 126.3093872, -142.7711029, 137.2509766, -268.6561279, 269.0805054
7: -143.5983124, 120.8984909, -156.0240479, 131.3002472, -274.8985596, 276.9225159
8: -172.2507935, 116.8822861, -187.1016083, 126.9695969, -299.2203979, 303.9838867
9: -130.3676605, 128.7661438, -141.6262207, 139.8688965, -270.2365112, 270.3923035

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2313198, upper bound: 326.2299142
time: 10.57 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2319624, upper bound: 326.2305816
time: 11.03 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -136.1607513, 108.2626038, -154.5318451, 122.7941513, -258.9548950, 262.7944031
1: -114.4997635, 96.3799362, -129.8892365, 109.3161087, -223.8158722, 226.2691650
2: -150.2152252, 98.5672913, -170.4293976, 111.5795059, -261.7947388, 268.9966431
3: -159.6873016, 84.4762192, -181.2545471, 95.6923065, -255.3796082, 265.7307739
4: -145.7973938, 112.3521423, -165.5770874, 127.4266205, -273.2239380, 277.9291992
5: -130.9673767, 102.2618027, -148.6919403, 115.9826508, -246.9500122, 250.9537354
6: -125.7639313, 120.8747559, -142.6577148, 137.1065521, -262.8704834, 263.5324707
7: -137.4401398, 115.7630844, -155.8712463, 131.1942444, -268.6343689, 271.6343384
8: -164.8439331, 111.7825470, -186.8654938, 126.7040787, -291.5480042, 298.6480408
9: -124.7931213, 123.2323456, -141.5217133, 139.7076111, -264.5007324, 264.7540588

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2305980, upper bound: 326.2293510
time: 9.60 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2312080, upper bound: 326.2299133
time: 10.25 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -161.7297363, 128.5123291, -162.9401093, 129.4762421, -291.2059937, 291.4524231
1: -136.0354462, 114.4629593, -137.0109406, 115.3046646, -251.3401184, 251.4739075
2: -178.4585266, 116.8729706, -179.7624359, 117.6828079, -296.1412964, 296.6354065
3: -189.7063141, 100.3253784, -191.1466980, 101.0220718, -290.7283936, 291.4720154
4: -173.3099518, 133.4069672, -174.6478271, 134.3875732, -307.6974182, 308.0547180
5: -155.5846405, 121.4575806, -156.7863007, 122.3480453, -277.9326477, 278.2438354
6: -149.2652740, 143.5444946, -150.3810883, 144.5945129, -293.8597717, 293.9255981
7: -163.1894836, 137.2744141, -164.3703308, 138.2780762, -301.4674988, 301.6447449
8: -195.6456604, 132.8266907, -197.0714722, 133.7678680, -329.4135132, 329.8981628
9: -148.0723877, 146.2958221, -149.1693726, 147.3383179, -295.4107056, 295.4652100

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2420881, upper bound: 326.2416762
time: 10.55 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2421852, upper bound: 326.2419090
time: 10.76 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -156.1790466, 124.1086273, -163.3908081, 129.8112793, -285.9903259, 287.4994202
1: -131.3665009, 110.5552368, -137.3584747, 115.5922394, -246.9587097, 247.9137115
2: -172.3339844, 112.9056625, -180.2224274, 117.9336624, -290.2676392, 293.1280823
3: -183.2108917, 96.8749542, -191.6634064, 101.1868973, -284.3977661, 288.5383606
4: -167.3520355, 128.8416595, -175.1183014, 134.7310944, -302.0831299, 303.9599304
5: -150.2592621, 117.3000031, -157.2312927, 122.6422348, -272.9014587, 274.5312805
6: -144.1638641, 138.6306610, -150.8025055, 144.9665222, -289.1303711, 289.4331665
7: -157.6241150, 132.6305695, -164.8053131, 138.6628113, -296.2869263, 297.4358215
8: -188.9472656, 128.2134857, -197.5411530, 133.9872437, -322.9343872, 325.7546387
9: -143.0314026, 141.2934113, -149.5953674, 147.7017822, -290.7331848, 290.8887939

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2408285, upper bound: 326.2406931
time: 7.80 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2409414, upper bound: 326.2409414
time: 8.69 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 18.16 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.16
Output dim: 7, lower bound: -326.2313198, upper bound: 326.2299142
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.16
Output dim: 7, lower bound: -326.2319624, upper bound: 326.2305816
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 18.16
Output dim: 7, lower bound: -326.2305980, upper bound: 326.2293510
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 18.16
Output dim: 7, lower bound: -326.2312080, upper bound: 326.2299133
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.16
Output dim: 7, lower bound: -326.2420881, upper bound: 326.2416762
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.16
Output dim: 7, lower bound: -326.2421852, upper bound: 326.2419090
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 18.16
Output dim: 7, lower bound: -326.2408285, upper bound: 326.2406931
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 18.16
Output dim: 7, lower bound: -326.2409414, upper bound: 326.2409414

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -124.0687943, 98.7027969, -148.8050842, 118.2768936, -242.3456879, 247.5078735
1: -104.2529221, 87.7876587, -125.0727234, 105.2827301, -209.5356293, 212.8603668
2: -136.8399200, 89.8894272, -164.1306000, 107.5374985, -244.3773956, 254.0200195
3: -145.4129791, 76.9778595, -174.5189514, 92.2495193, -237.6625061, 251.4968109
4: -132.8044739, 102.3337326, -159.4425507, 122.7118378, -255.5162659, 261.7762451
5: -119.2794037, 93.2870407, -143.1473846, 111.7623901, -231.0417938, 236.4344177
6: -114.6156158, 110.1389999, -137.3785858, 132.0554810, -246.6710968, 247.5175629
7: -125.2594376, 105.5472336, -150.1258545, 126.3604279, -251.6198425, 255.6730499
8: -150.2403259, 101.7768631, -180.0321198, 122.0966034, -272.3369141, 281.8089905
9: -113.7346497, 112.3136520, -136.2789917, 134.5776062, -248.3122559, 248.5926514

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2292658, upper bound: 326.2283769
time: 9.68 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2292658, upper bound: 326.2299142
time: 8.29 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -129.3818512, 102.9075775, -149.6256256, 118.9302826, -248.3120880, 252.5331879
1: -108.7970276, 91.5903702, -125.7914429, 105.8862076, -214.6832275, 217.3818054
2: -142.7381134, 93.7465439, -165.0567780, 108.1589355, -250.8970490, 258.8033142
3: -151.7253723, 80.2872543, -175.5078888, 92.7743530, -244.4997253, 255.7950745
4: -138.4878998, 106.7289581, -160.3121185, 123.4014664, -261.8893433, 267.0410767
5: -124.4038467, 97.2407455, -143.9460907, 112.3753128, -236.7791595, 241.1868286
6: -119.5115128, 114.8719330, -138.1346436, 132.7904510, -252.3019562, 253.0065613
7: -130.6390533, 110.0803757, -150.9711151, 127.0788574, -257.7178955, 261.0514832
8: -156.6698761, 106.1434097, -181.0260315, 122.7834091, -279.4532776, 287.1694336
9: -118.6095657, 117.1096115, -137.0416107, 135.3236389, -253.9331970, 254.1512146

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2298285, upper bound: 326.2290070
time: 11.52 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2298285, upper bound: 326.2290070
time: 11.64 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -117.8445892, 93.7658386, -148.5436096, 118.0501022, -235.8946838, 242.3094482
1: -99.0137024, 83.4022369, -124.8179855, 105.0648575, -204.0785522, 208.2201996
2: -129.9712219, 85.4374847, -163.8013916, 107.2700272, -237.2412415, 249.2388763
3: -138.1312103, 73.1047287, -174.2029266, 91.9595490, -230.0907440, 247.3076477
4: -126.1229324, 97.2119980, -159.1474915, 122.4656982, -248.5886230, 256.3594360
5: -113.3079987, 88.6209183, -142.9109039, 111.5197754, -224.8277588, 231.5318298
6: -108.8946915, 104.6282425, -137.1430359, 131.7937317, -240.6884155, 241.7712708
7: -119.0147018, 100.3383331, -149.8388519, 126.1433945, -245.1580963, 250.1771545
8: -142.7307434, 96.6050186, -179.6350098, 121.7192001, -264.4499512, 276.2400208
9: -108.0804749, 106.7009583, -136.0535431, 134.2941284, -242.3745728, 242.7544708

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2269179, upper bound: 326.2267145
time: 9.40 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2269179, upper bound: 326.2293510
time: 9.14 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -123.3572006, 98.1255493, -149.6159363, 118.9004059, -242.2575989, 247.7414551
1: -103.7257690, 87.3447189, -125.7460327, 105.8434525, -209.5691833, 213.0907593
2: -136.0858459, 89.4354630, -165.0023499, 108.0710983, -244.1569519, 254.4378052
3: -144.6770172, 76.5345535, -175.4848328, 92.6410904, -237.3181152, 252.0193787
4: -132.0195923, 101.7726898, -160.2883911, 123.3634644, -255.3830414, 262.0610962
5: -118.6227570, 92.7221680, -143.9484711, 112.3208084, -230.9435577, 236.6706390
6: -113.9719162, 109.5341492, -138.1298676, 132.7488098, -246.7207336, 247.6640167
7: -124.5925827, 105.0368500, -150.9353943, 127.0713043, -251.6638641, 255.9722443
8: -149.3970947, 101.1354141, -180.9324341, 122.6144028, -272.0115051, 282.0678101
9: -113.1361237, 111.6754456, -137.0440216, 135.2674255, -248.4035492, 248.7194672

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2273289, upper bound: 326.2273290
time: 8.44 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2273289, upper bound: 326.2299133
time: 8.17 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -143.3457642, 113.9606628, -157.0511169, 124.8095551, -268.1553345, 271.0117798
1: -120.4919662, 101.4392624, -132.0202484, 111.1224823, -231.6144104, 233.4595032
2: -158.1345215, 103.6980438, -173.2413330, 113.4423218, -271.5768127, 276.9393616
3: -168.0810547, 88.9168930, -184.2115173, 97.3531494, -265.4341736, 273.1284180
4: -153.5654449, 118.2145615, -168.3241577, 129.5083923, -283.0738525, 286.5387268
5: -137.8626556, 107.7664337, -151.1006317, 117.9581070, -255.8207703, 258.8670654
6: -132.3356781, 127.2372284, -144.9570465, 139.3682709, -271.7039490, 272.1942749
7: -144.6944733, 121.7951431, -158.4364471, 133.3083191, -278.0028076, 280.2315979
8: -173.4474030, 117.5924683, -189.9615326, 128.8669434, -302.3142700, 307.5539856
9: -131.3022919, 129.7003937, -143.7897034, 142.0136414, -273.3159180, 273.4900818

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2330869, upper bound: 326.2333000
time: 9.44 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2330869, upper bound: 326.2416762
time: 10.99 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -148.8181915, 118.2856674, -157.8800812, 125.4679337, -274.2861023, 276.1657104
1: -125.1644363, 105.3517303, -132.7479095, 111.7336807, -236.8981018, 238.0996399
2: -164.2105560, 107.6597137, -174.1767273, 114.0708160, -278.2813721, 281.8364258
3: -174.5621948, 92.3201218, -185.2109070, 97.8847504, -272.4469604, 277.5310059
4: -159.4170380, 122.7416534, -169.2050934, 130.2078400, -289.6248779, 291.9467468
5: -143.1331329, 111.8319244, -151.9064941, 118.5771561, -261.7102966, 263.7384033
6: -137.3733063, 132.1069336, -145.7209625, 140.1109619, -277.4842529, 277.8278809
7: -150.2296753, 126.4512482, -159.2913666, 134.0345764, -284.2642212, 285.7425842
8: -180.0674438, 122.0949631, -190.9653168, 129.5612946, -309.6286621, 313.0602722
9: -136.3146820, 134.6428833, -144.5623016, 142.7695465, -279.0842285, 279.2052002

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2333939, upper bound: 326.2337134
time: 11.06 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2333940, upper bound: 326.2419090
time: 10.45 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -137.7020569, 109.4841232, -157.3389435, 125.0155792, -262.7175903, 266.8230286
1: -115.7449722, 97.4664536, -132.2304688, 111.2958145, -227.0407867, 229.6969299
2: -151.9116211, 99.6677628, -173.5226746, 113.5764542, -265.4880371, 273.1904297
3: -161.4774017, 85.4064255, -184.5379181, 97.4160233, -258.8934326, 269.9443054
4: -147.5085297, 113.5730286, -168.6201935, 129.7179718, -277.2264404, 282.1932068
5: -132.4479980, 103.5405731, -151.3885040, 118.1316757, -250.5796814, 254.9290771
6: -127.1495819, 122.2421188, -145.2292633, 139.5960236, -266.7455750, 267.4713745
7: -139.0403748, 117.0773468, -158.7082825, 133.5582428, -272.5986328, 275.7856445
8: -166.6389465, 112.9002991, -190.2334900, 128.9497070, -295.5886536, 303.1337891
9: -126.1781540, 124.6155701, -144.0691986, 142.2299194, -268.4080811, 268.6847534

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2295037, upper bound: 326.2305381
time: 9.45 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2295037, upper bound: 326.2406931
time: 9.89 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -143.3927002, 113.9832382, -158.4607544, 125.9060287, -269.2986755, 272.4440002
1: -120.6022720, 101.5324707, -133.2017975, 112.1096420, -232.7118988, 234.7342682
2: -158.2253418, 103.7808914, -174.7789612, 114.4114685, -272.6368103, 278.5598450
3: -168.2112885, 88.9468002, -185.8764801, 98.1282578, -266.3395386, 274.8232727
4: -153.5945435, 118.2783356, -169.8145447, 130.6564178, -284.2509460, 288.0928650
5: -137.9282074, 107.7701187, -152.4743500, 118.9699097, -256.8980713, 260.2444458
6: -132.3869019, 127.3044662, -146.2615967, 140.5966187, -272.9835205, 273.5660706
7: -144.7894592, 121.9134750, -159.8540497, 134.5246124, -279.3140259, 281.7675171
8: -173.5205383, 117.5863800, -191.5907745, 129.8862762, -303.4067078, 309.1771545
9: -131.3876648, 129.7552185, -145.1025848, 143.2492981, -274.6369629, 274.8577881

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2299133, upper bound: 326.2312080
time: 11.39 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2295037, upper bound: 326.2409413
time: 10.02 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 22.79 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 7, lower bound: -326.2292658, upper bound: 326.2283769
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 7, lower bound: -326.2292658, upper bound: 326.2299142
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 7, lower bound: -326.2298285, upper bound: 326.2290070
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 7, lower bound: -326.2298285, upper bound: 326.2290070
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 7, lower bound: -326.2269179, upper bound: 326.2267145
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 7, lower bound: -326.2269179, upper bound: 326.2293510
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 7, lower bound: -326.2273289, upper bound: 326.2273290
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 7, lower bound: -326.2273289, upper bound: 326.2299133
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 7, lower bound: -326.2330869, upper bound: 326.2333000
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 7, lower bound: -326.2330869, upper bound: 326.2416762
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 7, lower bound: -326.2333939, upper bound: 326.2337134
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 7, lower bound: -326.2333940, upper bound: 326.2419090
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 7, lower bound: -326.2295037, upper bound: 326.2305381
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 7, lower bound: -326.2295037, upper bound: 326.2406931
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 7, lower bound: -326.2299133, upper bound: 326.2312080
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 7, lower bound: -326.2295037, upper bound: 326.2409413

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -124.0687943, 98.7027969, -128.6149292, 102.2801819, -226.3489685, 227.3177185
1: -104.2529221, 87.7876587, -108.1005020, 91.0181351, -195.2710114, 195.8881226
2: -136.8399200, 89.8894272, -141.8621674, 93.1243668, -229.9642792, 231.7515869
3: -145.4129791, 76.9778595, -150.8154602, 79.7663803, -225.1793518, 227.7933197
4: -132.8044739, 102.3337326, -137.7022247, 106.0977097, -238.9021912, 240.0359344
5: -119.2794037, 93.2870407, -123.6875534, 96.6194839, -215.8988953, 216.9745941
6: -114.6156158, 110.1389999, -118.8142319, 114.1759338, -228.7915497, 228.9532166
7: -125.2594376, 105.5472336, -129.8328857, 109.3937988, -234.6532288, 235.3801117
8: -150.2403259, 101.7768631, -155.7371521, 105.4953461, -255.7356720, 257.5140076
9: -113.7346497, 112.3136520, -117.9022980, 116.4004211, -230.1350708, 230.2159424

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2292658, upper bound: 326.2283769
time: 10.21 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2292658, upper bound: 326.2283769
time: 9.82 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -124.0687943, 98.7027969, -147.4209595, 117.1658249, -241.2346191, 246.1237488
1: -104.2529221, 87.7876587, -123.9408798, 104.3355484, -208.5884705, 211.7285309
2: -136.8399200, 89.8894272, -162.6346436, 106.5868607, -243.4267731, 252.5240784
3: -145.4129791, 76.9778595, -172.9245148, 91.4089432, -236.8219299, 249.9023743
4: -132.8044739, 102.3337326, -157.9574738, 121.5891342, -254.3936157, 260.2911987
5: -119.2794037, 93.2870407, -141.8181000, 110.7438431, -230.0232391, 235.1051331
6: -114.6156158, 110.1389999, -136.0986938, 130.8561249, -245.4717407, 246.2376709
7: -125.2594376, 105.5472336, -148.7876892, 125.2407379, -250.5001831, 254.3348846
8: -150.2403259, 101.7768631, -178.3774109, 120.9239044, -271.1642456, 280.1542358
9: -113.7346497, 112.3136520, -135.0357056, 133.3642273, -247.0988770, 247.3493652

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2292658, upper bound: 326.2283769
time: 8.90 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2292658, upper bound: 326.2299142
time: 8.14 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -129.3818512, 102.9075775, -129.3757629, 102.8895340, -232.2713623, 232.2833405
1: -108.7970276, 91.5903702, -108.7720184, 91.5812912, -200.3783264, 200.3623810
2: -142.7381134, 93.7465439, -142.7228699, 93.7048111, -236.4429321, 236.4694214
3: -151.7253723, 80.2872543, -151.7378387, 80.2570648, -231.9824219, 232.0250854
4: -138.4878998, 106.7289581, -138.5113220, 106.7394257, -245.2273254, 245.2402649
5: -124.4038467, 97.2407455, -124.4324265, 97.1909561, -221.5948029, 221.6731720
6: -119.5115128, 114.8719330, -119.5167923, 114.8624039, -234.3738861, 234.3887177
7: -130.6390533, 110.0803757, -130.6211395, 110.0655518, -240.7045746, 240.7015076
8: -156.6698761, 106.1434097, -156.6633148, 106.1329651, -262.8027954, 262.8067322
9: -118.6095657, 117.1096115, -118.6116409, 117.0968781, -235.7064209, 235.7212524

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2298285, upper bound: 326.2290070
time: 10.53 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2298285, upper bound: 326.2290070
time: 10.48 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -129.3818512, 102.9075775, -148.2700195, 117.8392639, -247.2210999, 251.1775818
1: -108.7970276, 91.5903702, -124.6848526, 104.9598541, -213.7568817, 216.2752228
2: -142.7381134, 93.7465439, -163.5930634, 107.2282333, -249.9663391, 257.3395996
3: -151.7253723, 80.2872543, -173.9449768, 91.9516373, -243.6770020, 254.2321930
4: -138.4878998, 106.7289581, -158.8597717, 122.3044434, -260.7923584, 265.5886841
5: -124.4038467, 97.2407455, -142.6430359, 111.3761520, -235.7799988, 239.8837891
6: -119.5115128, 114.8719330, -136.8806915, 131.6179047, -251.1294098, 251.7526093
7: -130.6390533, 110.0803757, -149.6620789, 125.9821701, -256.6212158, 259.7424622
8: -156.6698761, 106.1434097, -179.4056396, 121.6332321, -278.3031006, 285.5490417
9: -118.6095657, 117.1096115, -135.8229523, 134.1385040, -252.7480774, 252.9325562

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2298285, upper bound: 326.2290070
time: 10.11 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2298285, upper bound: 326.2290070
time: 10.28 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -117.8445892, 93.7658386, -127.4337234, 101.3231583, -219.1677551, 221.1995544
1: -99.0137024, 83.4022369, -107.0676422, 90.1475372, -189.1612244, 190.4698639
2: -129.9712219, 85.4374847, -140.5164032, 92.1906891, -222.1618958, 225.9538879
3: -138.1312103, 73.1047287, -149.4241180, 78.9005966, -217.0317841, 222.5288391
4: -126.1229324, 97.2119980, -136.4215393, 105.0924759, -231.2154083, 233.6335144
5: -113.3079987, 88.6209183, -122.5672073, 95.6815491, -208.9895325, 211.1881256
6: -108.8946915, 104.6282425, -117.7335434, 113.1022186, -221.9969177, 222.3617859
7: -119.0147018, 100.3383331, -128.6167755, 108.4010696, -227.4157715, 228.9550934
8: -142.7307434, 96.6050186, -154.2342072, 104.3551559, -247.0859070, 250.8392334
9: -108.0804749, 106.7009583, -116.8366623, 115.2858582, -223.3663025, 223.5375824

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1787427, upper bound: 326.1809026
time: 8.77 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1712306, upper bound: 326.1708203
time: 6.16 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -117.8445892, 93.7658386, -148.0631866, 117.6555099, -235.5000916, 241.8290253
1: -99.0137024, 83.4022369, -124.4496155, 104.7584610, -203.7721558, 207.8518524
2: -129.9712219, 85.4374847, -163.3100586, 106.9739227, -236.9451294, 248.7475433
3: -138.1312103, 73.1047287, -173.6628113, 91.6916428, -229.8228455, 246.7675476
4: -126.1229324, 97.2119980, -158.6332550, 122.0862122, -248.2091370, 255.8452301
5: -113.3079987, 88.6209183, -142.4483032, 111.1809921, -224.4889526, 231.0692139
6: -108.8946915, 104.6282425, -136.6981354, 131.3980713, -240.2927551, 241.3263702
7: -119.0147018, 100.3383331, -149.4155273, 125.7853546, -244.8000488, 249.7538300
8: -142.7307434, 96.6050186, -179.0755920, 121.2982788, -264.0290222, 275.6806030
9: -108.0804749, 106.7009583, -135.6318817, 133.8994598, -241.9799042, 242.3328094

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1787427, upper bound: 326.1809026
time: 10.09 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1712306, upper bound: 326.1784496
time: 7.93 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -123.3572006, 98.1255493, -128.4598541, 102.1375580, -225.4947357, 226.5853882
1: -103.7257690, 87.3447189, -107.9631577, 90.8963547, -194.6221161, 195.3078613
2: -136.0858459, 89.4354630, -141.6671753, 92.9606552, -229.0464783, 231.1026306
3: -144.6770172, 76.5345535, -150.6568604, 79.5560684, -224.2330933, 227.1914062
4: -132.0195923, 101.7726898, -137.5125732, 105.9542236, -237.9738159, 239.2852478
5: -118.6227570, 92.7221680, -123.5645599, 96.4492035, -215.0719604, 216.2867279
6: -113.9719162, 109.5341492, -118.6794815, 114.0195618, -227.9914856, 228.2136230
7: -124.5925827, 105.0368500, -129.6703644, 109.2938995, -233.8864594, 234.7071991
8: -149.3970947, 101.1354141, -155.4779510, 105.2123413, -254.6094360, 256.6133728
9: -113.1361237, 111.6754456, -117.7872162, 116.2186050, -229.3547363, 229.4626617

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2242171, upper bound: 326.2246509
time: 10.01 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2236484, upper bound: 326.2236484
time: 7.89 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -123.3572006, 98.1255493, -149.1845551, 118.5475159, -241.9047241, 247.3100739
1: -103.7257690, 87.3447189, -125.4233627, 105.5735321, -209.2992859, 212.7680664
2: -136.0858459, 89.4354630, -164.5667572, 107.8062897, -243.8921356, 254.0022278
3: -144.6770172, 76.5345535, -174.9989471, 92.4072952, -237.0843201, 251.5335083
4: -132.0195923, 101.7726898, -159.8290863, 123.0235596, -255.0431519, 261.6017761
5: -118.6227570, 92.7221680, -143.5338898, 112.0198975, -230.6426544, 236.2560577
6: -113.9719162, 109.5341492, -137.7294159, 132.4007416, -246.3726501, 247.2635651
7: -124.5925827, 105.0368500, -150.5612183, 126.7507935, -251.3433685, 255.5980530
8: -149.3970947, 101.1354141, -180.4341888, 122.2379608, -271.6350708, 281.5695801
9: -113.1361237, 111.6754456, -136.6650848, 134.9217224, -248.0578461, 248.3405304

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2242171, upper bound: 326.2267377
time: 9.93 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2236484, upper bound: 326.2260644
time: 9.06 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -143.3457642, 113.9606628, -128.6149292, 102.2801819, -245.6259460, 242.5755920
1: -120.4919662, 101.4392624, -108.1005020, 91.0181351, -211.5100708, 209.5397186
2: -158.1345215, 103.6980438, -141.8621674, 93.1243668, -251.2588806, 245.5602112
3: -168.0810547, 88.9168930, -150.8154602, 79.7663803, -247.8474274, 239.7323151
4: -153.5654449, 118.2145615, -137.7022247, 106.0977097, -259.6631470, 255.9167633
5: -137.8626556, 107.7664337, -123.6875534, 96.6194839, -234.4821472, 231.4539795
6: -132.3356781, 127.2372284, -118.8142319, 114.1759338, -246.5116119, 246.0514526
7: -144.6944733, 121.7951431, -129.8328857, 109.3937988, -254.0882568, 251.6280212
8: -173.4474030, 117.5924683, -155.7371521, 105.4953461, -278.9426880, 273.3296204
9: -131.3022919, 129.7003937, -117.9022980, 116.4004211, -247.7026978, 247.6026917

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 175

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2330869, upper bound: 326.2333000
time: 10.93 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2330869, upper bound: 326.2333000
time: 9.97 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -143.3457642, 113.9606628, -147.4209595, 117.1658249, -260.5115967, 261.3815918
1: -120.4919662, 101.4392624, -123.9408798, 104.3355484, -224.8275146, 225.3801422
2: -158.1345215, 103.6980438, -162.6346436, 106.5868607, -264.7213745, 266.3326111
3: -168.0810547, 88.9168930, -172.9245148, 91.4089432, -259.4899902, 261.8414001
4: -153.5654449, 118.2145615, -157.9574738, 121.5891342, -275.1545715, 276.1720276
5: -137.8626556, 107.7664337, -141.8181000, 110.7438431, -248.6064911, 249.5845337
6: -132.3356781, 127.2372284, -136.0986938, 130.8561249, -263.1918030, 263.3359375
7: -144.6944733, 121.7951431, -148.7876892, 125.2407379, -269.9351807, 270.5828247
8: -173.4474030, 117.5924683, -178.3774109, 120.9239044, -294.3712769, 295.9698486
9: -131.3022919, 129.7003937, -135.0357056, 133.3642273, -264.6665039, 264.7360840

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 175

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2330869, upper bound: 326.2416762
time: 10.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2330869, upper bound: 326.2416762
time: 11.58 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -148.8181915, 118.2856674, -129.3757629, 102.8895340, -251.7077179, 247.6614227
1: -125.1644363, 105.3517303, -108.7720184, 91.5812912, -216.7457123, 214.1237488
2: -164.2105560, 107.6597137, -142.7228699, 93.7048111, -257.9153748, 250.3825836
3: -174.5621948, 92.3201218, -151.7378387, 80.2570648, -254.8192444, 244.0579529
4: -159.4170380, 122.7416534, -138.5113220, 106.7394257, -266.1564636, 261.2529602
5: -143.1331329, 111.8319244, -124.4324265, 97.1909561, -240.3240967, 236.2643433
6: -137.3733063, 132.1069336, -119.5167923, 114.8624039, -252.2356873, 251.6237183
7: -150.2296753, 126.4512482, -130.6211395, 110.0655518, -260.2952271, 257.0723877
8: -180.0674438, 122.0949631, -156.6633148, 106.1329651, -286.2003174, 278.7582703
9: -136.3146820, 134.6428833, -118.6116409, 117.0968781, -253.4115295, 253.2545166

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2333939, upper bound: 326.2337134
time: 10.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2333939, upper bound: 326.2337134
time: 10.28 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -148.8181915, 118.2856674, -148.2700195, 117.8392639, -266.6574707, 266.5556641
1: -125.1644363, 105.3517303, -124.6848526, 104.9598541, -230.1242676, 230.0365906
2: -164.2105560, 107.6597137, -163.5930634, 107.2282333, -271.4387817, 271.2527771
3: -174.5621948, 92.3201218, -173.9449768, 91.9516373, -266.5138245, 266.2651062
4: -159.4170380, 122.7416534, -158.8597717, 122.3044434, -281.7214966, 281.6013794
5: -143.1331329, 111.8319244, -142.6430359, 111.3761520, -254.5092773, 254.4749603
6: -137.3733063, 132.1069336, -136.8806915, 131.6179047, -268.9912109, 268.9876099
7: -150.2296753, 126.4512482, -149.6620789, 125.9821701, -276.2117920, 276.1132812
8: -180.0674438, 122.0949631, -179.4056396, 121.6332321, -301.7006531, 301.5005798
9: -136.3146820, 134.6428833, -135.8229523, 134.1385040, -270.4531860, 270.4658203

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2333940, upper bound: 326.2419090
time: 10.27 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2333940, upper bound: 326.2419090
time: 9.99 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -137.7020569, 109.4841232, -127.4337234, 101.3231583, -239.0252075, 236.9178314
1: -115.7449722, 97.4664536, -107.0676422, 90.1475372, -205.8925018, 204.5340881
2: -151.9116211, 99.6677628, -140.5164032, 92.1906891, -244.1023102, 240.1841583
3: -161.4774017, 85.4064255, -149.4241180, 78.9005966, -240.3779755, 234.8305206
4: -147.5085297, 113.5730286, -136.4215393, 105.0924759, -252.6010132, 249.9945679
5: -132.4479980, 103.5405731, -122.5672073, 95.6815491, -228.1295471, 226.1077728
6: -127.1495819, 122.2421188, -117.7335434, 113.1022186, -240.2518005, 239.9756317
7: -139.0403748, 117.0773468, -128.6167755, 108.4010696, -247.4414368, 245.6941223
8: -166.6389465, 112.9002991, -154.2342072, 104.3551559, -270.9941101, 267.1345215
9: -126.1781540, 124.6155701, -116.8366623, 115.2858582, -241.4640198, 241.4522400

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2263972, upper bound: 326.2278777
time: 10.97 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2254659, upper bound: 326.2263262
time: 10.46 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 23.19 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -326.2292658, upper bound: 326.2283769
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -326.2292658, upper bound: 326.2283769
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -326.2292658, upper bound: 326.2283769
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -326.2292658, upper bound: 326.2299142
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -326.2298285, upper bound: 326.2290070
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -326.2298285, upper bound: 326.2290070
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -326.2298285, upper bound: 326.2290070
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -326.2298285, upper bound: 326.2290070
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -326.1787427, upper bound: 326.1809026
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 23.19
Output dim: 7, lower bound: -326.1712306, upper bound: 326.1708203
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -326.1787427, upper bound: 326.1809026
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -326.1712306, upper bound: 326.1784496
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -326.2242171, upper bound: 326.2246509
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -326.2236484, upper bound: 326.2236484
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -326.2242171, upper bound: 326.2267377
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -326.2236484, upper bound: 326.2260644
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -326.2330869, upper bound: 326.2333000
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -326.2330869, upper bound: 326.2333000
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -326.2330869, upper bound: 326.2416762
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -326.2330869, upper bound: 326.2416762
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -326.2333939, upper bound: 326.2337134
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -326.2333939, upper bound: 326.2337134
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -326.2333940, upper bound: 326.2419090
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -326.2333940, upper bound: 326.2419090
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -326.2263972, upper bound: 326.2278777
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -326.2254659, upper bound: 326.2263262
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.19
Output dim: 7, lower bound: -326.2295037, upper bound: 326.2406931
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.19
Output dim: 7, lower bound: -326.2299133, upper bound: 326.2312080
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.19
Output dim: 7, lower bound: -326.2295037, upper bound: 326.2409413
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=328.3682861328125
rel_dist={7: [-326.2560128858547, 326.2560128858547]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2466022, upper bound: 326.2464614
time: 13.13 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2523184, upper bound: 326.2523184
time: 12.57 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 25.89 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 25.89
Output dim: 7, lower bound: -326.2466022, upper bound: 326.2464614
IS_A2, status: Status.UNKNOWN, split count: 1, time: 25.89
Output dim: 7, lower bound: -326.2523184, upper bound: 326.2523184

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -142.9952087, 113.6855774, -152.0590820, 120.8632584, -263.8584290, 265.7446594
1: -120.2532425, 101.1967163, -127.8664780, 107.5958176, -227.8490601, 229.0631714
2: -157.7607422, 103.4563980, -167.7504120, 109.9189911, -267.6796875, 271.2068176
3: -167.6863556, 88.7308502, -178.3235931, 94.3293457, -262.0156860, 267.0544434
4: -153.1332703, 117.9750061, -162.8888092, 125.4253998, -278.5586548, 280.8638000
5: -137.5261078, 107.3854294, -146.2578125, 114.1766739, -251.7027740, 253.6432343
6: -132.0471954, 126.9280167, -140.3809814, 134.9504700, -266.9976196, 267.3089905
7: -144.2999115, 121.4841537, -153.4026947, 129.0964050, -273.3963013, 274.8868408
8: -173.0928497, 117.4631882, -183.9933167, 124.9122314, -298.0050659, 301.4565125
9: -131.0026245, 129.3968506, -139.2514343, 137.5523834, -268.5549622, 268.6482849

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 41

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2333848, upper bound: 326.2328802
time: 15.41 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2331232, upper bound: 326.2326725
time: 10.67 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -162.4753876, 129.1040802, -166.2585754, 132.1083679, -294.5837402, 295.3626709
1: -136.6650543, 114.9896698, -139.8399811, 117.6561966, -254.3212585, 254.8296051
2: -179.2824554, 117.4086914, -183.4491272, 120.1024780, -299.3849487, 300.8578186
3: -190.5786285, 100.7902908, -195.0155334, 103.1247406, -293.7033081, 295.8057861
4: -174.1093292, 134.0215607, -178.1817474, 137.1342010, -311.2435303, 312.2033081
5: -156.3003693, 122.0177612, -159.9475403, 124.8523712, -281.1527405, 281.9653015
6: -149.9510803, 144.2054901, -153.4319305, 147.5489044, -297.4999695, 297.6374207
7: -163.9395142, 137.9003448, -167.7294159, 141.0720673, -305.0115967, 305.6297607
8: -196.5453186, 133.4469147, -201.0963593, 136.5697784, -333.1151123, 334.5432739
9: -148.7507019, 146.9695892, -152.1924591, 150.3676605, -299.1183472, 299.1619873

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2445150, upper bound: 326.2444016
time: 11.44 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2440465, upper bound: 326.2440463
time: 11.07 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 23.86 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 23.86
Output dim: 7, lower bound: -326.2333848, upper bound: 326.2328802
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 23.86
Output dim: 7, lower bound: -326.2331232, upper bound: 326.2326725
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 23.86
Output dim: 7, lower bound: -326.2445150, upper bound: 326.2444016
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 23.86
Output dim: 7, lower bound: -326.2440465, upper bound: 326.2440463

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -137.4821472, 109.3115158, -143.3946533, 113.9886475, -251.4707947, 252.7061462
1: -115.6020355, 97.3036728, -120.5535202, 101.4767685, -217.0787811, 217.8571777
2: -151.6726074, 99.4996490, -158.1834869, 103.6993866, -255.3719940, 257.6831360
3: -161.2369690, 85.2941055, -168.1862488, 88.9282837, -250.1652527, 253.4803314
4: -147.2218933, 113.4316788, -153.5980377, 118.2849808, -265.5068665, 267.0296936
5: -132.2350769, 103.2462540, -137.9409485, 107.6705246, -239.9055939, 241.1871948
6: -126.9773254, 122.0431366, -132.4136505, 127.2724915, -254.2498169, 254.4567413
7: -138.7602386, 116.8597183, -144.6952209, 121.8267670, -260.5870056, 261.5549316
8: -166.4428864, 112.8758240, -173.5440826, 117.7023315, -284.1452026, 286.4198914
9: -125.9891510, 124.4167252, -131.3710785, 129.7237854, -255.7129364, 255.7877960

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2281563, upper bound: 326.2275681
time: 12.20 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2288783, upper bound: 326.2282619
time: 11.37 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -130.5077820, 103.7771301, -142.8351440, 113.5214996, -244.0292664, 246.6122589
1: -109.7419968, 92.3966827, -120.0463562, 101.0433426, -210.7853088, 212.4430237
2: -143.9760590, 94.5235825, -157.5185852, 103.2136612, -247.1897278, 252.0421600
3: -153.0744781, 80.9587860, -167.5178375, 88.4490433, -241.5235138, 248.4765930
4: -139.7285309, 107.7010574, -152.9807892, 117.7904205, -257.5188904, 260.6817322
5: -125.5426102, 98.0239868, -137.4156036, 107.1973419, -232.7399445, 235.4395905
6: -120.5668564, 115.8683853, -131.9012299, 126.7454453, -247.3123016, 247.7695923
7: -131.7686310, 111.0315933, -144.1048889, 121.3560181, -253.1246033, 255.1364746
8: -158.0219269, 107.0853500, -172.7827148, 117.0759583, -275.0979004, 279.8680420
9: -119.6575851, 118.1355209, -130.8720703, 129.1702118, -248.8277740, 249.0075989

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2279081, upper bound: 326.2273715
time: 10.94 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2286228, upper bound: 326.2280385
time: 11.04 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -156.5812683, 124.4270477, -157.1333160, 124.8669586, -281.4482422, 281.5603333
1: -131.6896210, 110.8267593, -132.1372833, 111.2113876, -242.9010010, 242.9640503
2: -172.7694702, 113.1743088, -173.3659668, 113.5459290, -286.3153992, 286.5402832
3: -183.6837769, 97.1162033, -184.3399048, 97.4368210, -281.1206055, 281.4560852
4: -167.7899323, 129.1634827, -168.3985138, 129.6118774, -297.4017944, 297.5619812
5: -150.6430359, 117.5898132, -151.1893768, 117.9963684, -268.6394043, 268.7791748
6: -144.5299835, 138.9812622, -145.0391998, 139.4611664, -283.9911499, 284.0204163
7: -158.0113068, 132.9523621, -158.5512695, 133.4102173, -291.4215088, 291.5036316
8: -189.4340973, 128.5444641, -190.0870667, 128.9783020, -318.4124146, 318.6315308
9: -143.3888397, 141.6442566, -143.8891907, 142.1222229, -285.5110474, 285.5334473

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2410546, upper bound: 326.2409045
time: 11.89 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2412562, upper bound: 326.2411494
time: 12.84 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -150.9845428, 119.9878845, -157.7974548, 125.3716202, -276.3561707, 277.7853394
1: -126.9947052, 106.8976974, -132.6646881, 111.6498489, -238.6445160, 239.5623779
2: -166.6012573, 109.1900101, -174.0617676, 113.9478531, -280.5491028, 283.2517700
3: -177.1332550, 93.6475830, -185.1040344, 97.7369232, -274.8701782, 278.7516174
4: -161.7762299, 124.5674057, -169.0955505, 130.1288147, -291.9050293, 293.6629333
5: -145.2756042, 113.4068756, -151.8399353, 118.4504166, -263.7259216, 265.2468262
6: -139.3886566, 134.0301056, -145.6571045, 140.0225220, -279.4111633, 279.6871948
7: -152.4132538, 128.2827301, -159.1997528, 133.9726105, -286.3858643, 287.4823914
8: -182.6769562, 123.8959503, -190.8123474, 129.3729706, -312.0499268, 314.7081909
9: -138.3119354, 136.6116943, -144.5063782, 142.6777344, -280.9896240, 281.1180725

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2406160, upper bound: 326.2405551
time: 12.89 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2407724, upper bound: 326.2407724
time: 14.13 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 28.73 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.73
Output dim: 7, lower bound: -326.2281563, upper bound: 326.2275681
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.73
Output dim: 7, lower bound: -326.2288783, upper bound: 326.2282619
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.73
Output dim: 7, lower bound: -326.2279081, upper bound: 326.2273715
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.73
Output dim: 7, lower bound: -326.2286228, upper bound: 326.2280385
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.73
Output dim: 7, lower bound: -326.2410546, upper bound: 326.2409045
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.73
Output dim: 7, lower bound: -326.2412562, upper bound: 326.2411494
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.73
Output dim: 7, lower bound: -326.2406160, upper bound: 326.2405551
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.73
Output dim: 7, lower bound: -326.2407724, upper bound: 326.2407724

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -119.3455582, 94.9548035, -132.1924744, 105.1104889, -224.4559937, 227.1472473
1: -100.2677536, 84.4530640, -111.0652618, 93.5202789, -193.7879944, 195.5183258
2: -131.6226196, 86.4999084, -145.7793579, 95.6365509, -227.2591705, 232.2792664
3: -139.8873444, 74.0337601, -154.9892578, 81.9457855, -221.8331299, 229.0230103
4: -127.7387619, 98.4418259, -141.5669861, 109.0028534, -236.7416077, 240.0087891
5: -114.7464752, 89.7411804, -127.1214371, 99.3198929, -214.0663452, 216.8626099
6: -110.2727585, 105.9531403, -122.0965729, 117.3295593, -227.6022949, 228.0496826
7: -120.5126190, 101.5859451, -133.4078827, 112.3776627, -232.8902893, 234.9938202
8: -144.5421906, 97.8472748, -160.0166321, 108.3786545, -252.9208374, 257.8638916
9: -109.4401321, 108.0467529, -121.1400375, 119.5977707, -229.0379028, 229.1867828

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1752832, upper bound: 326.1739710
time: 14.99 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1742341, upper bound: 326.1731343
time: 12.52 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -124.5168991, 99.0478592, -133.6649475, 106.2875900, -230.8044891, 232.7127838
1: -104.6919403, 88.1548996, -112.3646317, 94.6095123, -199.3014526, 200.5195007
2: -137.3646545, 90.2554245, -147.4437714, 96.7612228, -234.1258698, 237.6991882
3: -146.0324554, 77.2543640, -156.7730560, 82.8943176, -228.9267273, 234.0274048
4: -133.2715302, 102.7182159, -143.1306152, 110.2441940, -243.5157166, 245.8488159
5: -119.7347260, 93.5889664, -128.5592346, 100.4233017, -220.1580200, 222.1481934
6: -115.0376663, 110.5612335, -123.4544983, 118.6548157, -233.6924438, 234.0157318
7: -125.7507324, 106.0001984, -134.9317017, 113.6753616, -239.4260864, 240.9318848
8: -150.8009491, 102.0946045, -161.8045502, 109.6126175, -260.4135437, 263.8991699
9: -114.1853867, 112.7142334, -122.5130463, 120.9419098, -235.1272888, 235.2272797

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1783431, upper bound: 326.1770378
time: 13.05 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1773049, upper bound: 326.1761970
time: 13.38 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -112.1125336, 89.2186813, -131.2981873, 104.3815765, -216.4941101, 220.5168457
1: -94.1889954, 79.3640671, -110.2730179, 92.8517609, -187.0407562, 189.6370850
2: -123.6459885, 81.3380280, -144.7500763, 94.9092941, -218.5552826, 226.0881042
3: -131.4268799, 69.5379181, -153.9309082, 81.2566986, -212.6835785, 223.4688263
4: -119.9692841, 92.4958038, -140.5931854, 108.2310257, -228.2003174, 233.0889893
5: -107.8070145, 84.3249741, -126.2762756, 98.5985565, -206.4055634, 210.6012573
6: -103.6253128, 99.5527649, -121.2776108, 116.5080261, -220.1333008, 220.8303680
7: -113.2648163, 95.5416870, -132.4815216, 111.6261902, -224.8910065, 228.0231934
8: -135.8150177, 91.8417435, -158.8541565, 107.4717484, -243.2867737, 250.6958923
9: -102.8749390, 101.5319290, -120.3366623, 118.7404633, -221.6154022, 221.8685913

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1750166, upper bound: 326.1737222
time: 13.07 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1740902, upper bound: 326.1729782
time: 9.88 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -117.8034821, 93.7166519, -133.3255157, 105.9889832, -223.7924652, 227.0421448
1: -99.0511703, 83.4315872, -112.0359802, 94.3283234, -193.3794861, 195.4675598
2: -129.9555817, 85.4624176, -147.0221863, 96.4275208, -226.3830719, 232.4846039
3: -138.1784515, 73.0775299, -156.3605194, 82.5481110, -220.7265320, 229.4380493
4: -126.0558243, 97.2032776, -142.7476501, 109.9303589, -235.9861755, 239.9509277
5: -113.2927704, 88.5574036, -128.2399139, 100.1128311, -213.4055939, 216.7973175
6: -108.8666229, 104.6144714, -123.1421356, 118.3173599, -227.1839752, 227.7565765
7: -119.0210571, 100.3886261, -134.5590363, 113.3843689, -232.4054260, 234.9476624
8: -142.6943817, 96.5212173, -161.3076630, 109.1654587, -251.8598328, 257.8288879
9: -108.0915833, 106.6674271, -122.2110214, 120.5825958, -228.6741791, 228.8784332

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1780738, upper bound: 326.1768113
time: 10.81 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1771461, upper bound: 326.1760607
time: 9.38 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -138.3208771, 109.9731064, -145.7988586, 115.8869019, -254.2077789, 255.7719574
1: -116.2523651, 97.8906250, -122.5382767, 103.1643829, -219.4167328, 220.4288940
2: -152.5829926, 100.0881805, -160.8169861, 105.3908691, -257.9738464, 260.9051514
3: -162.2024231, 85.7848969, -170.9944153, 90.3771439, -252.5795593, 256.7792969
4: -148.1771545, 114.0737076, -156.2242889, 120.2238235, -268.4009094, 270.2980042
5: -133.0399780, 103.9913025, -140.2470703, 109.5486526, -242.5886230, 244.2383728
6: -127.7147980, 122.7839737, -134.6008759, 129.4033813, -257.1181335, 257.3847961
7: -139.6401367, 117.5786896, -147.1311035, 123.8525238, -263.4926758, 264.7097778
8: -167.3858643, 113.4145584, -176.4016724, 119.5497818, -286.9355774, 289.8161316
9: -126.7320175, 125.1598282, -133.5408020, 131.8748627, -258.6068420, 258.7005615

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2367996, upper bound: 326.2364238
time: 13.55 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2364069, upper bound: 326.2361714
time: 11.18 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -143.6373749, 114.1742706, -147.4215546, 117.1739807, -260.8113098, 261.5958252
1: -120.7910461, 101.6920624, -123.9597092, 104.3576126, -225.1486359, 225.6517639
2: -158.4858856, 103.9384689, -162.6474457, 106.6168365, -265.1027222, 266.5859070
3: -168.5021667, 89.0897446, -172.9489899, 91.4154434, -259.9176025, 262.0387268
4: -153.8616791, 118.4709930, -157.9488983, 121.5910263, -275.4526978, 276.4198914
5: -138.1600494, 107.9403229, -141.8228607, 110.7578735, -248.9179230, 249.7631836
6: -132.6087189, 127.5147629, -136.0954895, 130.8580627, -263.4667053, 263.6102600
7: -145.0195465, 122.1031494, -148.8035431, 125.2700424, -270.2895813, 270.9066772
8: -173.8162842, 117.7845154, -178.3679810, 120.9059219, -294.7221985, 296.1524048
9: -131.6011353, 129.9609528, -135.0462952, 133.3560181, -264.9571533, 265.0072021

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2256104, upper bound: 326.2370657
time: 14.14 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2369352, upper bound: 326.2368064
time: 21.43 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -132.4463959, 105.3134537, -146.1057434, 116.1082916, -248.5546875, 251.4191742
1: -111.3213577, 93.7653732, -122.7612000, 103.3496094, -214.6709595, 216.5265656
2: -146.1138153, 95.9083328, -161.1240845, 105.5373306, -251.6511536, 257.0324097
3: -155.3280945, 82.1372147, -171.3392792, 90.4480515, -245.7761230, 253.4764862
4: -141.8656311, 109.2484894, -156.5401611, 120.4457932, -262.3114014, 265.7886353
5: -127.4048080, 99.6016235, -140.5516357, 109.7357254, -237.1405029, 240.1532593
6: -122.3182755, 117.5879898, -134.8916473, 129.6476440, -251.9658966, 252.4796295
7: -133.7699890, 112.6800766, -147.4251251, 124.1167068, -257.8866882, 260.1051941
8: -160.2968140, 108.5302505, -176.6951447, 119.6415329, -279.9383545, 285.2254028
9: -121.4036407, 119.8762817, -133.8335724, 132.1063538, -253.5099945, 253.7098236

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2216054, upper bound: 326.2212183
time: 14.34 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2203581, upper bound: 326.2203011
time: 10.42 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -138.2725983, 109.9237747, -148.2839050, 117.8390656, -256.1115417, 258.2076721
1: -116.2940292, 97.9261475, -124.6502686, 104.9324722, -221.2264557, 222.5764160
2: -152.5764313, 100.1191635, -163.5632019, 107.1559982, -259.7324219, 263.6823120
3: -162.2191467, 85.7659454, -173.9379425, 91.8361816, -254.0553284, 259.7038879
4: -148.0992737, 114.0642090, -158.8611145, 122.2651749, -270.3644104, 272.9253235
5: -133.0166626, 103.9353485, -142.6615601, 111.3644180, -244.3810730, 246.5969086
6: -127.6811981, 122.7709045, -136.8958282, 131.5924835, -259.2736816, 259.6667175
7: -139.6539307, 117.6290359, -149.6488647, 125.9939270, -265.6478577, 267.2778625
8: -167.3425903, 113.3311081, -179.3326263, 121.4612122, -288.8037720, 292.6637268
9: -126.7359009, 125.1424103, -135.8390656, 134.0901794, -260.8260803, 260.9814758

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2369943, upper bound: 326.2367741
time: 12.03 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.2365348, upper bound: 326.2365348
time: 10.87 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.21 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.21
Output dim: 7, lower bound: -326.1752832, upper bound: 326.1739710
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.21
Output dim: 7, lower bound: -326.1742341, upper bound: 326.1731343
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.21
Output dim: 7, lower bound: -326.1783431, upper bound: 326.1770378
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.21
Output dim: 7, lower bound: -326.1773049, upper bound: 326.1761970
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.21
Output dim: 7, lower bound: -326.1750166, upper bound: 326.1737222
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.21
Output dim: 7, lower bound: -326.1740902, upper bound: 326.1729782
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.21
Output dim: 7, lower bound: -326.1780738, upper bound: 326.1768113
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.21
Output dim: 7, lower bound: -326.1771461, upper bound: 326.1760607
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.21
Output dim: 7, lower bound: -326.2367996, upper bound: 326.2364238
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.21
Output dim: 7, lower bound: -326.2364069, upper bound: 326.2361714
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.21
Output dim: 7, lower bound: -326.2256104, upper bound: 326.2370657
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.21
Output dim: 7, lower bound: -326.2369352, upper bound: 326.2368064
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.21
Output dim: 7, lower bound: -326.2216054, upper bound: 326.2212183
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.21
Output dim: 7, lower bound: -326.2203581, upper bound: 326.2203011
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.21
Output dim: 7, lower bound: -326.2369943, upper bound: 326.2367741
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.21
Output dim: 7, lower bound: -326.2365348, upper bound: 326.2365348

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -114.5506668, 91.1430664, -124.3358459, 98.8665085, -213.4171753, 215.4788971
1: -96.2294540, 81.0653458, -104.4482422, 87.9687500, -184.1981964, 185.5135803
2: -126.3229828, 83.0906677, -137.0984344, 90.0501633, -216.3731384, 220.1890869
3: -134.2807770, 71.0701828, -145.8022461, 77.0906754, -211.3714600, 216.8724365
4: -122.5836792, 94.4830627, -133.1202087, 102.5159302, -225.0996094, 227.6032715
5: -110.1377716, 86.1480103, -119.5694733, 93.4327927, -203.5705566, 205.7174835
6: -105.8665543, 101.6959076, -114.8777008, 110.3543320, -216.2208710, 216.5736084
7: -115.7098389, 97.5755463, -125.5394669, 105.8058395, -221.5156860, 223.1150208
8: -138.7453766, 93.8688049, -150.5211487, 101.8615265, -240.6069031, 244.3899536
9: -105.0866852, 103.7297134, -114.0049286, 112.5252838, -217.6119690, 217.7346497

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1718863, upper bound: 326.1708320
time: 11.82 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1718699, upper bound: 326.1707183
time: 13.83 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -99.5012054, 79.1872330, -112.0904007, 89.1040115, -188.6052246, 191.2776337
1: -83.5822601, 70.4492188, -94.0913010, 79.2700119, -162.8522644, 164.5405273
2: -109.7288361, 72.3947601, -123.5752411, 81.2507477, -190.9795837, 195.9700012
3: -116.7126312, 61.8071327, -131.5519867, 69.4900589, -186.2026672, 193.3591156
4: -106.3951874, 82.0590668, -119.9459839, 92.2930908, -198.6882782, 202.0050507
5: -95.6724396, 74.8703384, -107.7724686, 84.1656570, -179.8380890, 182.6427917
6: -92.0350494, 88.3776703, -103.6202850, 99.5165787, -191.5516205, 191.9979553
7: -100.6721649, 85.0057907, -113.2858200, 95.5811691, -196.2533264, 198.2916107
8: -120.5836563, 81.3956070, -135.7016907, 91.5582962, -212.1419525, 217.0972900
9: -91.4376907, 90.2111816, -102.9172821, 101.5169525, -192.9546356, 193.1284637

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1709439, upper bound: 326.1700644
time: 11.21 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1709234, upper bound: 326.1699442
time: 12.48 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -119.7192078, 95.2357635, -125.8393555, 100.0697708, -219.7889709, 221.0750885
1: -100.6496658, 84.7645111, -105.7724991, 89.0794220, -189.7290955, 190.5370026
2: -132.0657806, 86.8440628, -138.8009186, 91.1962814, -223.2620544, 225.6449890
3: -140.4243774, 74.2898254, -147.6248169, 78.0595551, -218.4839020, 221.9146423
4: -128.1145325, 98.7561722, -134.7180328, 103.7835922, -231.8981323, 233.4742126
5: -115.1242447, 89.9949188, -121.0395432, 94.5601044, -209.6843567, 211.0344543
6: -110.6283722, 106.3033066, -116.2632446, 111.7098694, -222.3382263, 222.5665436
7: -120.9481201, 101.9880981, -127.0973129, 107.1314087, -228.0795135, 229.0854187
8: -145.0042114, 98.1154022, -152.3511810, 103.1223450, -248.1265564, 250.4665680
9: -109.8287354, 108.3954010, -115.4066391, 113.8988419, -223.7275543, 223.8020172

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1749296, upper bound: 326.1738449
time: 11.12 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1749293, upper bound: 326.1737286
time: 12.11 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -104.6510315, 83.2643661, -113.6179581, 90.3208466, -194.9718628, 196.8823242
1: -87.9864655, 74.1341476, -95.4328842, 80.3941956, -168.3806610, 169.5670319
2: -115.4451904, 76.1285172, -125.2962799, 82.4095154, -197.8547058, 201.4248047
3: -122.8282471, 65.0107574, -133.3950806, 70.4681320, -193.2963867, 198.4058380
4: -111.9055862, 86.3152847, -121.5687408, 93.5766678, -205.4822235, 207.8840332
5: -100.6365356, 78.7001114, -109.2588043, 85.3049622, -185.9414825, 187.9589233
6: -96.7804489, 92.9608841, -105.0249100, 100.8865891, -197.6670227, 197.9857941
7: -105.8846283, 89.3974457, -114.8575821, 96.9170609, -202.8016815, 204.2550354
8: -126.8139267, 85.6258087, -137.5522461, 92.8349991, -219.6489258, 223.1780396
9: -96.1585693, 94.8544235, -104.3330460, 102.9037933, -199.0623627, 199.1874695

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1740106, upper bound: 326.1731040
time: 11.54 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1740044, upper bound: 326.1730175
time: 10.25 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -107.1612015, 85.2825089, -123.1015320, 97.8654709, -205.0266418, 208.3840332
1: -90.0162659, 75.8646774, -103.3634644, 87.0575867, -177.0738525, 179.2281494
2: -118.1728134, 77.8162155, -135.6921997, 89.0762177, -207.2490234, 213.5084076
3: -125.6383896, 66.4777603, -144.3486176, 76.1891174, -201.8274994, 210.8263702
4: -114.6457443, 88.4067154, -131.7837677, 101.4612045, -216.1069489, 220.1904907
5: -103.0493088, 80.6139374, -118.3970642, 92.4544754, -195.5037842, 199.0110016
6: -99.0760040, 95.1561279, -113.7467651, 109.2306824, -208.3066864, 208.9028931
7: -108.3050842, 91.4005203, -124.2723618, 104.7693710, -213.0744629, 215.6728516
8: -129.8297119, 87.7310028, -148.9488678, 100.6676788, -230.4973755, 236.6798706
9: -98.3780746, 97.0717850, -112.8931122, 111.3567047, -209.7347717, 209.9648743

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1717369, upper bound: 326.1707033
time: 9.05 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1717274, upper bound: 326.1705969
time: 11.86 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -92.8791351, 73.9363785, -111.8585281, 88.9055481, -181.7846375, 185.7948914
1: -78.0217743, 65.7946243, -93.8648834, 79.0788345, -157.1006165, 159.6595001
2: -102.4264908, 67.6686478, -123.2818680, 81.0056229, -183.4321136, 190.9505005
3: -108.9637909, 57.6870956, -131.2541504, 69.2173309, -178.1810913, 188.9412384
4: -99.2749023, 76.6195297, -119.6783295, 92.0746078, -191.3494873, 196.2978516
5: -89.3184052, 69.9149628, -107.5585938, 83.9526901, -173.2710876, 177.4735565
6: -85.9474258, 82.5173569, -103.3973389, 99.2837219, -185.2311096, 185.9147034
7: -94.0336227, 79.4735794, -113.0178375, 95.3844986, -189.4181213, 192.4914246
8: -112.5901947, 75.8987122, -135.3352966, 91.2131577, -203.8033447, 211.2340088
9: -85.4251328, 84.2479782, -102.7109680, 101.2647171, -186.6898346, 186.9589539

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1708774, upper bound: 326.1699983
time: 13.40 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -326.1708595, upper bound: 326.1698861
time: 10.60 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -112.9062042, 89.8241806, -125.1839523, 99.5174484, -212.4236450, 215.0081329
1: -94.9244156, 79.9707642, -105.1742706, 88.5742950, -183.4987183, 185.1450348
2: -124.5457840, 81.9804306, -138.0289001, 90.6345520, -215.1803284, 220.0093384
3: -132.4539337, 70.0516815, -146.8453369, 77.5167923, -209.9707031, 216.8970184
4: -120.7909470, 93.1585312, -133.9964600, 103.2072830, -223.9982147, 227.1549683
5: -108.5872192, 84.8879089, -120.4161453, 94.0112686, -202.5984497, 205.3040314
6: -104.3671188, 100.2669601, -115.6626892, 111.0898209, -215.4569397, 215.9296570
7: -114.1180496, 96.2940674, -126.4073486, 106.5746765, -220.6927032, 222.7014160
8: -136.7779999, 92.4563217, -151.4705658, 102.4080658, -239.1860504, 243.9268799
9: -103.6447601, 102.2577591, -114.8181305, 113.2507935, -216.8955078, 217.0758362

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1747533, upper bound: 326.1737045
time: 13.66 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1747486, upper bound: 326.1735979
time: 13.01 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -98.5132523, 78.3908463, -113.8013458, 90.4447098, -188.9579620, 192.1921692
1: -82.8325272, 69.8186417, -95.5503464, 80.4882584, -163.3207855, 165.3689880
2: -108.6741409, 71.7468338, -125.4571762, 82.4554138, -191.1295471, 197.2040100
3: -115.6424408, 61.1926003, -133.5749969, 70.4513474, -186.0937805, 194.7675934
4: -105.3038101, 81.2742767, -121.7439117, 93.6966171, -199.0003967, 203.0181580
5: -94.7440643, 74.1020355, -109.4405136, 85.4007263, -180.1447906, 183.5425415
6: -91.1345901, 87.5266571, -105.1840973, 101.0144806, -192.1490784, 192.7107391
7: -99.7321396, 84.2677612, -115.0052490, 97.0644989, -196.7966309, 199.2730103
8: -119.4031830, 80.5299683, -137.6843567, 92.8275986, -212.2307587, 218.2143250
9: -90.5856781, 89.3283005, -104.4974670, 103.0225754, -193.6082458, 193.8257751

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 175

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1739110, upper bound: 326.1730361
time: 10.86 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -326.1739019, upper bound: 326.1729435
time: 11.44 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 23.91 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 23.91
Output dim: 7, lower bound: -326.1718863, upper bound: 326.1708320
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 23.91
Output dim: 7, lower bound: -326.1718699, upper bound: 326.1707183
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 23.91
Output dim: 7, lower bound: -326.1709439, upper bound: 326.1700644
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 23.91
Output dim: 7, lower bound: -326.1709234, upper bound: 326.1699442
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.91
Output dim: 7, lower bound: -326.1749296, upper bound: 326.1738449
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.91
Output dim: 7, lower bound: -326.1749293, upper bound: 326.1737286
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.91
Output dim: 7, lower bound: -326.1740106, upper bound: 326.1731040
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.91
Output dim: 7, lower bound: -326.1740044, upper bound: 326.1730175
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 23.91
Output dim: 7, lower bound: -326.1717369, upper bound: 326.1707033
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 23.91
Output dim: 7, lower bound: -326.1717274, upper bound: 326.1705969
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 23.91
Output dim: 7, lower bound: -326.1708774, upper bound: 326.1699983
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 23.91
Output dim: 7, lower bound: -326.1708595, upper bound: 326.1698861
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.91
Output dim: 7, lower bound: -326.1747533, upper bound: 326.1737045
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.91
Output dim: 7, lower bound: -326.1747486, upper bound: 326.1735979
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.91
Output dim: 7, lower bound: -326.1739110, upper bound: 326.1730361
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.91
Output dim: 7, lower bound: -326.1739019, upper bound: 326.1729435
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.91
Output dim: 7, lower bound: -326.2367996, upper bound: 326.2364238
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.91
Output dim: 7, lower bound: -326.2364069, upper bound: 326.2361714
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.91
Output dim: 7, lower bound: -326.2256104, upper bound: 326.2370657
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.91
Output dim: 7, lower bound: -326.2369352, upper bound: 326.2368064
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.91
Output dim: 7, lower bound: -326.2216054, upper bound: 326.2212183
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.91
Output dim: 7, lower bound: -326.2203581, upper bound: 326.2203011
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.91
Output dim: 7, lower bound: -326.2369943, upper bound: 326.2367741
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.91
Output dim: 7, lower bound: -326.2365348, upper bound: 326.2365348
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=328.3682861328125
rel_dist={7: [-326.25584232239004, 326.2558422835341]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1824.20 seconds
