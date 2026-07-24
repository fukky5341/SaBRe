## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.603988435


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=44, inp2_unstable=44, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.3055841, 1.0798980, 0.3055841, 1.0798980, -0.7743139, 0.7743139)
1: (-0.3411395, 0.3320597, -0.3411395, 0.3320597, -0.6731992, 0.6731992)
2: (-0.2506143, 0.4382139, -0.2506143, 0.4382139, -0.6888281, 0.6888281)
3: (-0.2480090, 0.3303499, -0.2480090, 0.3303499, -0.5783589, 0.5783589)
4: (-0.3511734, 0.3201319, -0.3511734, 0.3201319, -0.6713052, 0.6713052)
5: (-0.3786278, 0.4931411, -0.3786278, 0.4931411, -0.8717690, 0.8717690)
6: (-0.2745771, 0.3690816, -0.2745771, 0.3690816, -0.6436587, 0.6436587)
7: (-0.3625940, 0.3785164, -0.3625940, 0.3785164, -0.7411104, 0.7411104)
8: (-0.3543938, 0.4575191, -0.3543938, 0.4575191, -0.8119130, 0.8119130)
9: (-0.3473027, 0.4444523, -0.3473027, 0.4444523, -0.7917550, 0.7917550)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.72 + 2.71 = 4.43 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.6367709, upper bound: 0.6367709

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6367709, upper bound: 0.6367589
time: 1.12 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6367589, upper bound: 0.6367589
time: 1.13 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.49 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.49
Output dim: 0, lower bound: -0.6367709, upper bound: 0.6367589
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.49
Output dim: 0, lower bound: -0.6367589, upper bound: 0.6367589

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.3683689, 1.0707364, 0.3186938, 1.0778632, -0.7094944, 0.7520426
1: -0.3036801, 0.2960106, -0.3332010, 0.3236491, -0.6273292, 0.6292115
2: -0.2167687, 0.3959269, -0.2427635, 0.4288427, -0.6456114, 0.6386904
3: -0.2211120, 0.2974810, -0.2421756, 0.3236980, -0.5448101, 0.5396566
4: -0.3152697, 0.2790599, -0.3420209, 0.3103288, -0.6255985, 0.6210808
5: -0.3394672, 0.4441904, -0.3696832, 0.4831411, -0.8226082, 0.8138735
6: -0.2426672, 0.3288350, -0.2680584, 0.3590735, -0.6017407, 0.5968934
7: -0.3279970, 0.3375294, -0.3553840, 0.3696299, -0.6976269, 0.6929134
8: -0.3154933, 0.4096564, -0.3445019, 0.4470982, -0.7625915, 0.7541583
9: -0.3086374, 0.3972074, -0.3384192, 0.4347338, -0.7433712, 0.7356266

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=44, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 111

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6320343, upper bound: 0.6318955
time: 1.17 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6320343, upper bound: 0.6320194
time: 1.14 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.3188961, 1.0778323, 0.3104303, 1.0791470, -0.7602508, 0.7674021
1: -0.3330703, 0.3235391, -0.3382217, 0.3289182, -0.6619885, 0.6617608
2: -0.2426414, 0.4287103, -0.2477141, 0.4347497, -0.6773911, 0.6764244
3: -0.2420906, 0.3235913, -0.2458550, 0.3279003, -0.5699909, 0.5694463
4: -0.3419145, 0.3101675, -0.3477528, 0.3165289, -0.6584433, 0.6579204
5: -0.3695494, 0.4829842, -0.3753247, 0.4894571, -0.8590065, 0.8583089
6: -0.2679562, 0.3589429, -0.2721757, 0.3653475, -0.6333036, 0.6311187
7: -0.3552704, 0.3694969, -0.3599382, 0.3752104, -0.7304808, 0.7294351
8: -0.3443795, 0.4469347, -0.3507197, 0.4536606, -0.7980400, 0.7976544
9: -0.3382865, 0.4345843, -0.3439946, 0.4408709, -0.7791574, 0.7785789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=44, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=31, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 111

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6320194, upper bound: 0.6318952
time: 1.20 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6320194, upper bound: 0.6320194
time: 1.11 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.06 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.06
Output dim: 0, lower bound: -0.6320343, upper bound: 0.6318955
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.06
Output dim: 0, lower bound: -0.6320343, upper bound: 0.6320194
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.06
Output dim: 0, lower bound: -0.6320194, upper bound: 0.6318952
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.06
Output dim: 0, lower bound: -0.6320194, upper bound: 0.6320194

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.3914783, 1.0675311, 0.4228431, 1.0631808, -0.6717025, 0.6446880
1: -0.2901993, 0.2831264, -0.2719029, 0.2656400, -0.5558393, 0.5550293
2: -0.2048494, 0.3806432, -0.1886722, 0.3598999, -0.5647493, 0.5693154
3: -0.2113051, 0.2853724, -0.1979951, 0.2689383, -0.4802434, 0.4833676
4: -0.3027993, 0.2649731, -0.2858741, 0.2458541, -0.5486535, 0.5508472
5: -0.3255255, 0.4261181, -0.3066033, 0.4015903, -0.7271158, 0.7327214
6: -0.2308894, 0.3148461, -0.2149042, 0.2958601, -0.5267495, 0.5297503
7: -0.3152769, 0.3228190, -0.2980126, 0.3028538, -0.6181307, 0.6208315
8: -0.3023083, 0.3923563, -0.2844132, 0.3688765, -0.6711847, 0.6767695
9: -0.2949350, 0.3797139, -0.2763380, 0.3559712, -0.6509062, 0.6560519

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6320343, upper bound: 0.6318087
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6319784, upper bound: 0.6318096
time: 1.21 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.4192349, 1.0636811, 0.3856308, 1.0683420, -0.6491070, 0.6780503
1: -0.2740076, 0.2676516, -0.2936104, 0.2863866, -0.5603942, 0.5612620
2: -0.1905331, 0.3622861, -0.2078653, 0.3845106, -0.5750437, 0.5701513
3: -0.1995264, 0.2708288, -0.2137867, 0.2884363, -0.4879627, 0.4846155
4: -0.2878211, 0.2480535, -0.3059548, 0.2685375, -0.5563586, 0.5540082
5: -0.3087802, 0.4044120, -0.3290531, 0.4306911, -0.7394713, 0.7334651
6: -0.2167431, 0.2980443, -0.2338695, 0.3183858, -0.5351289, 0.5319138
7: -0.2999986, 0.3051505, -0.3184956, 0.3265413, -0.6265399, 0.6236461
8: -0.2864718, 0.3715777, -0.3056445, 0.3967338, -0.6832057, 0.6772221
9: -0.2784774, 0.3587025, -0.2984022, 0.3841402, -0.6626176, 0.6571047

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6320343, upper bound: 0.6319660
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6319784, upper bound: 0.6319660
time: 1.07 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.3426952, 1.0742974, 0.4153038, 1.0642263, -0.7215311, 0.6589936
1: -0.3186567, 0.3103240, -0.2763008, 0.2698432, -0.5884999, 0.5866249
2: -0.2300105, 0.4129065, -0.1925607, 0.3648862, -0.5948967, 0.6054673
3: -0.2320070, 0.3109332, -0.2011945, 0.2728887, -0.5048956, 0.5121278
4: -0.3291239, 0.2947095, -0.2899425, 0.2504498, -0.5795737, 0.5846520
5: -0.3549557, 0.4642675, -0.3111517, 0.4074864, -0.7624421, 0.7754192
6: -0.2557518, 0.3443761, -0.2187466, 0.3004239, -0.5561757, 0.5631227
7: -0.3421287, 0.3538719, -0.3021625, 0.3076529, -0.6497815, 0.6560344
8: -0.3301414, 0.4288759, -0.2887148, 0.3745205, -0.7046618, 0.7175907
9: -0.3238598, 0.4166418, -0.2808083, 0.3616783, -0.6855381, 0.6974500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6320194, upper bound: 0.6318088
time: 1.14 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6319660, upper bound: 0.6318096
time: 7.76 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.3690327, 1.0706444, 0.3789096, 1.0692744, -0.7002418, 0.6917347
1: -0.3032929, 0.2956405, -0.2975312, 0.2901338, -0.5934267, 0.5931717
2: -0.2164263, 0.3954880, -0.2113320, 0.3889557, -0.6053820, 0.6068200
3: -0.2208304, 0.2971333, -0.2166389, 0.2919580, -0.5127884, 0.5137721
4: -0.3149116, 0.2786551, -0.3095817, 0.2726344, -0.5875460, 0.5882368
5: -0.3390665, 0.4436713, -0.3331080, 0.4359473, -0.7750138, 0.7767793
6: -0.2423289, 0.3284333, -0.2372949, 0.3224544, -0.5647833, 0.5657282
7: -0.3276317, 0.3371068, -0.3221951, 0.3308197, -0.6584514, 0.6593019
8: -0.3151147, 0.4091593, -0.3094794, 0.4017654, -0.7168801, 0.7186387
9: -0.3082435, 0.3967047, -0.3023872, 0.3892281, -0.6974717, 0.6990919

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=43, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6320194, upper bound: 0.6319660
time: 1.18 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6319660, upper bound: 0.6319660
time: 1.09 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.12 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 0, lower bound: -0.6320343, upper bound: 0.6318087
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 0, lower bound: -0.6319784, upper bound: 0.6318096
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 0, lower bound: -0.6320343, upper bound: 0.6319660
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 0, lower bound: -0.6319784, upper bound: 0.6319660
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 0, lower bound: -0.6320194, upper bound: 0.6318088
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 0, lower bound: -0.6319660, upper bound: 0.6318096
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 0, lower bound: -0.6320194, upper bound: 0.6319660
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 0, lower bound: -0.6319660, upper bound: 0.6319660

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.4238990, 1.0630343, 0.4342785, 1.0615945, -0.6376954, 0.6287558
1: -0.2712869, 0.2650512, -0.2652321, 0.2592646, -0.5305515, 0.5302834
2: -0.1881276, 0.3592016, -0.1827742, 0.3523371, -0.5404648, 0.5419757
3: -0.1975470, 0.2683851, -0.1931424, 0.2629465, -0.4604936, 0.4615276
4: -0.2853043, 0.2452105, -0.2797034, 0.2388835, -0.5241878, 0.5249139
5: -0.3059665, 0.4007646, -0.2997046, 0.3926478, -0.6986142, 0.7004693
6: -0.2143661, 0.2952210, -0.2090762, 0.2889380, -0.5033041, 0.5042971
7: -0.2974314, 0.3021818, -0.2917182, 0.2955745, -0.5930060, 0.5939000
8: -0.2838109, 0.3680862, -0.2778889, 0.3603159, -0.6441268, 0.6459751
9: -0.2757120, 0.3551721, -0.2695577, 0.3473150, -0.6230270, 0.6247298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=43, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6246871
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6226884
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.3710127, 1.0703698, 0.4341432, 1.0616132, -0.6906005, 0.6362267
1: -0.3021379, 0.2945365, -0.2653110, 0.2593399, -0.5614777, 0.5598475
2: -0.2154051, 0.3941784, -0.1828438, 0.3524264, -0.5678315, 0.5770223
3: -0.2199901, 0.2960958, -0.1931999, 0.2630174, -0.4830074, 0.4892957
4: -0.3138431, 0.2774482, -0.2797763, 0.2389659, -0.5528090, 0.5572245
5: -0.3378720, 0.4421228, -0.2997862, 0.3927535, -0.7306255, 0.7419090
6: -0.2413197, 0.3272346, -0.2091451, 0.2890199, -0.5303396, 0.5363797
7: -0.3265419, 0.3358465, -0.2917925, 0.2956607, -0.6222026, 0.6276390
8: -0.3139849, 0.4076772, -0.2779661, 0.3604172, -0.6744022, 0.6856433
9: -0.3070697, 0.3952060, -0.2696379, 0.3474174, -0.6544870, 0.6648439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=43, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6246661
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6226865
time: 1.13 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.4499373, 1.0594225, 0.3975630, 1.0666871, -0.6167499, 0.6618595
1: -0.2560976, 0.2505343, -0.2866499, 0.2797343, -0.5358318, 0.5371842
2: -0.1746976, 0.3419810, -0.2017111, 0.3766192, -0.5513169, 0.5436921
3: -0.1864974, 0.2547418, -0.2087232, 0.2821843, -0.4686817, 0.4634650
4: -0.2712533, 0.2293384, -0.2995159, 0.2612641, -0.5325174, 0.5288544
5: -0.2902578, 0.3804024, -0.3218548, 0.4213601, -0.7116178, 0.7022573
6: -0.2010956, 0.2794591, -0.2277883, 0.3111630, -0.5122586, 0.5072474
7: -0.2830989, 0.2856070, -0.3119278, 0.3189460, -0.6020448, 0.5975347
8: -0.2689549, 0.3485935, -0.2988368, 0.3878015, -0.6567564, 0.6474303
9: -0.2602732, 0.3354615, -0.2913274, 0.3751079, -0.6353810, 0.6267889

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=43, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6249792
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6213716
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.4012553, 1.0661750, 0.3972527, 1.0667300, -0.6654747, 0.6689223
1: -0.2844959, 0.2776755, -0.2868309, 0.2799070, -0.5644029, 0.5645064
2: -0.1998067, 0.3741771, -0.2018711, 0.3768243, -0.5766310, 0.5760482
3: -0.2071562, 0.2802495, -0.2088548, 0.2823468, -0.4895030, 0.4891044
4: -0.2975233, 0.2590134, -0.2996833, 0.2614531, -0.5589764, 0.5586967
5: -0.3196271, 0.4184726, -0.3220418, 0.4216025, -0.7412297, 0.7405144
6: -0.2259065, 0.3089279, -0.2279464, 0.3113509, -0.5372573, 0.5368743
7: -0.3098952, 0.3165956, -0.3120984, 0.3191434, -0.6290386, 0.6286939
8: -0.2967301, 0.3850372, -0.2990137, 0.3880336, -0.6847637, 0.6840509
9: -0.2891380, 0.3723131, -0.2915111, 0.3753427, -0.6644807, 0.6638242

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=43, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6249534
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6213716
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.3768725, 1.0695570, 0.4268284, 1.0626280, -0.6857555, 0.6427286
1: -0.2987194, 0.2912695, -0.2695780, 0.2634180, -0.5621374, 0.5608475
2: -0.2123827, 0.3903030, -0.1866166, 0.3572641, -0.5696468, 0.5769196
3: -0.2175034, 0.2930253, -0.1963039, 0.2668501, -0.4843535, 0.4893292
4: -0.3106810, 0.2738762, -0.2837236, 0.2434249, -0.5541059, 0.5575998
5: -0.3343369, 0.4375402, -0.3041992, 0.3984739, -0.7328107, 0.7417394
6: -0.2383332, 0.3236875, -0.2128731, 0.2934476, -0.5317808, 0.5365606
7: -0.3233164, 0.3321164, -0.2958189, 0.3003169, -0.6236333, 0.6279353
8: -0.3106416, 0.4032905, -0.2821394, 0.3658931, -0.6765347, 0.6854299
9: -0.3035951, 0.3907703, -0.2739750, 0.3529544, -0.6565496, 0.6647453

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=43, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6249942
time: 1.13 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6227149
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.3261993, 1.0765857, 0.4266569, 1.0626515, -0.7364522, 0.6499287
1: -0.3282794, 0.3195210, -0.2696781, 0.2635137, -0.5917931, 0.5891991
2: -0.2385186, 0.4238162, -0.1867051, 0.3573776, -0.5958962, 0.6105213
3: -0.2390073, 0.3195766, -0.1963767, 0.2669400, -0.5059472, 0.5159533
4: -0.3380255, 0.3047650, -0.2838162, 0.2435294, -0.5815549, 0.5885811
5: -0.3649075, 0.4771675, -0.3043027, 0.3986080, -0.7635155, 0.7814703
6: -0.2641590, 0.3543614, -0.2129606, 0.2935514, -0.5577105, 0.5673220
7: -0.3512086, 0.3643725, -0.2959132, 0.3004262, -0.6516348, 0.6602857
8: -0.3395529, 0.4412247, -0.2822374, 0.3660214, -0.7055743, 0.7234621
9: -0.3336406, 0.4291289, -0.2740767, 0.3530843, -0.6867249, 0.7032056

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=43, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6249899
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6227146
time: 1.15 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.4013204, 1.0661659, 0.3909101, 1.0676099, -0.6662896, 0.6752558
1: -0.2844580, 0.2776394, -0.2905308, 0.2834432, -0.5679012, 0.5681702
2: -0.1997730, 0.3741342, -0.2051425, 0.3810191, -0.5807922, 0.5792767
3: -0.2071286, 0.2802155, -0.2115463, 0.2856701, -0.4927988, 0.4917618
4: -0.2974883, 0.2589738, -0.3031060, 0.2653195, -0.5628078, 0.5620798
5: -0.3195879, 0.4184215, -0.3258683, 0.4265627, -0.7461506, 0.7442898
6: -0.2258734, 0.3088883, -0.2311791, 0.3151902, -0.5410635, 0.5400674
7: -0.3098595, 0.3165541, -0.3155897, 0.3231808, -0.6330403, 0.6321437
8: -0.2966929, 0.3849885, -0.3026325, 0.3927818, -0.6894747, 0.6876211
9: -0.2890993, 0.3722636, -0.2952719, 0.3801440, -0.6692433, 0.6675354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=43, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6253811
time: 1.14 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6213716
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.3573002, 1.0722717, 0.3905443, 1.0676605, -0.7103602, 0.6817274
1: -0.3101369, 0.3021817, -0.2907442, 0.2836473, -0.5937843, 0.5929258
2: -0.2224777, 0.4032472, -0.2053311, 0.3812610, -0.6037387, 0.6085783
3: -0.2258093, 0.3032807, -0.2117016, 0.2858619, -0.5116712, 0.5149822
4: -0.3212426, 0.2858069, -0.3033034, 0.2655422, -0.5867848, 0.5891103
5: -0.3461447, 0.4528462, -0.3260890, 0.4268485, -0.7729932, 0.7789352
6: -0.2483084, 0.3355352, -0.2313654, 0.3154117, -0.5637200, 0.5669006
7: -0.3340897, 0.3445752, -0.3157910, 0.3234137, -0.6575034, 0.6603662
8: -0.3218085, 0.4179424, -0.3028413, 0.3930557, -0.7148643, 0.7207837
9: -0.3152001, 0.4055860, -0.2954888, 0.3804210, -0.6956211, 0.7010748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=43, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6253734
time: 1.12 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6213716
time: 5.88 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 8.90 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 8.90
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6246871
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 8.90
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6226884
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 8.90
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6246661
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 8.90
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6226865
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 8.90
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6249792
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 8.90
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6213716
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 8.90
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6249534
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 8.90
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6213716
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 8.90
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6249942
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 8.90
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6227149
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 8.90
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6249899
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 8.90
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6227146
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 8.90
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6253811
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 8.90
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6213716
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 8.90
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6253734
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 8.90
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6213716

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.4293997, 1.0622712, 0.4824845, 1.0549082, -0.6255084, 0.5797867
1: -0.2680779, 0.2619844, -0.2371112, 0.2323883, -0.5004662, 0.4990957
2: -0.1852903, 0.3555636, -0.1579106, 0.3204553, -0.5057456, 0.5134741
3: -0.1952127, 0.2655028, -0.1726854, 0.2376881, -0.4329008, 0.4381882
4: -0.2823359, 0.2418574, -0.2536900, 0.2094986, -0.4918346, 0.4955474
5: -0.3026479, 0.3964629, -0.2706224, 0.3549497, -0.6575976, 0.6670853
6: -0.2115626, 0.2918910, -0.1845077, 0.2597573, -0.4713199, 0.4763988
7: -0.2944035, 0.2986802, -0.2651837, 0.2648890, -0.5592925, 0.5638639
8: -0.2806723, 0.3639681, -0.2503850, 0.3242283, -0.6049006, 0.6143531
9: -0.2724504, 0.3510080, -0.2409750, 0.3108238, -0.5832742, 0.5919830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 111

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6246871
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6246871
time: 7.66 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.4824229, 1.0549166, 0.5427965, 1.0461988, -0.5637760, 0.5121201
1: -0.2371472, 0.2324229, -0.2019952, 0.1973477, -0.4344949, 0.4344181
2: -0.1579424, 0.3204962, -0.1285425, 0.2798682, -0.4378106, 0.4490387
3: -0.1727116, 0.2377205, -0.1460848, 0.2063776, -0.3790892, 0.3838053
4: -0.2537234, 0.2095363, -0.2201848, 0.1747467, -0.4284701, 0.4297211
5: -0.2706596, 0.3549980, -0.2319856, 0.3116806, -0.5823402, 0.5869836
6: -0.1845393, 0.2597946, -0.1560066, 0.2209897, -0.4055291, 0.4158012
7: -0.2652177, 0.2649283, -0.2326368, 0.2251036, -0.4903213, 0.4975652
8: -0.2504202, 0.3242746, -0.2149514, 0.2762848, -0.5267050, 0.5392260
9: -0.2410116, 0.3108706, -0.2043484, 0.2678843, -0.5088960, 0.5152190

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 111

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6226884
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6226884
time: 1.15 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.3766983, 1.0695812, 0.4823077, 1.0549325, -0.6782341, 0.5872734
1: -0.2988211, 0.2913665, -0.2372145, 0.2324869, -0.5313081, 0.5285809
2: -0.2124726, 0.3904181, -0.1580018, 0.3205723, -0.5330449, 0.5484198
3: -0.2175774, 0.2931167, -0.1727605, 0.2377808, -0.4553581, 0.4658772
4: -0.3107750, 0.2739824, -0.2537854, 0.2096065, -0.5203814, 0.5277678
5: -0.3344419, 0.4376765, -0.2707291, 0.3550880, -0.6895299, 0.7084057
6: -0.2384221, 0.3237929, -0.1845980, 0.2598642, -0.4982862, 0.5083908
7: -0.3234122, 0.3322273, -0.2652811, 0.2650016, -0.5884138, 0.5975084
8: -0.3107409, 0.4034207, -0.2504859, 0.3243607, -0.6351016, 0.6539066
9: -0.3036984, 0.3909021, -0.2410798, 0.3109577, -0.6146560, 0.6319819

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 111

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213626, upper bound: 0.6246661
time: 1.21 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213626, upper bound: 0.6246661
time: 1.46 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.4280194, 1.0624624, 0.5426989, 1.0462135, -0.6181941, 0.5197636
1: -0.2688832, 0.2627540, -0.2020521, 0.1974065, -0.4662897, 0.4648061
2: -0.1860024, 0.3564766, -0.1285868, 0.2799332, -0.4659356, 0.4850633
3: -0.1957985, 0.2662261, -0.1461299, 0.2064249, -0.4022234, 0.4123560
4: -0.2830807, 0.2426989, -0.2202396, 0.1747991, -0.4578798, 0.4629385
5: -0.3034806, 0.3975425, -0.2320528, 0.3117423, -0.6152229, 0.6295953
6: -0.2122661, 0.2927266, -0.1560463, 0.2210570, -0.4333231, 0.4487729
7: -0.2951634, 0.2995588, -0.2326884, 0.2251695, -0.5203329, 0.5322471
8: -0.2814599, 0.3650013, -0.2150097, 0.2763681, -0.5578280, 0.5800110
9: -0.2732688, 0.3520529, -0.2044079, 0.2679488, -0.5412176, 0.5564608

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 111

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213626, upper bound: 0.6226865
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213626, upper bound: 0.6226865
time: 1.20 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.4553882, 1.0586665, 0.4455253, 1.0600346, -0.6046464, 0.6131412
1: -0.2529178, 0.2474952, -0.2586713, 0.2529940, -0.5059118, 0.5061665
2: -0.1718862, 0.3383758, -0.1769733, 0.3448989, -0.5167851, 0.5153491
3: -0.1841841, 0.2518857, -0.1883696, 0.2570534, -0.4412375, 0.4402552
4: -0.2683120, 0.2260156, -0.2736342, 0.2320278, -0.5003398, 0.4996498
5: -0.2869693, 0.3761396, -0.2929194, 0.3838527, -0.6708219, 0.6690590
6: -0.1983175, 0.2761594, -0.2033442, 0.2821299, -0.4804475, 0.4795036
7: -0.2800985, 0.2821371, -0.2855275, 0.2884154, -0.5685140, 0.5676646
8: -0.2658446, 0.3445129, -0.2714720, 0.3518966, -0.6177412, 0.6159849
9: -0.2570412, 0.3313352, -0.2628891, 0.3388011, -0.5958424, 0.5942243

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6213716
time: 1.17 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6213716
time: 1.22 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.5096335, 1.0511423, 0.5190927, 1.0497622, -0.5401287, 0.5320497
1: -0.2212740, 0.2172521, -0.2157649, 0.2117040, -0.4329781, 0.4330170
2: -0.1439077, 0.3025001, -0.1392584, 0.2959188, -0.4398265, 0.4417585
3: -0.1611643, 0.2234629, -0.1570175, 0.2182487, -0.3794130, 0.3804803
4: -0.2390397, 0.1929496, -0.2336697, 0.1874491, -0.4264888, 0.4266194
5: -0.2542437, 0.3337188, -0.2482402, 0.3267536, -0.5809972, 0.5819590
6: -0.1706712, 0.2433231, -0.1659274, 0.2372991, -0.4079703, 0.4092506
7: -0.2502400, 0.2476072, -0.2451193, 0.2412727, -0.4915127, 0.4927266
8: -0.2348953, 0.3039044, -0.2292367, 0.2964547, -0.5313500, 0.5331411
9: -0.2248777, 0.2902725, -0.2189773, 0.2834705, -0.5083481, 0.5092498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6218435, upper bound: 0.6202619
time: 1.30 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6206941, upper bound: 0.6202619
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.4068679, 1.0653964, 0.4451512, 1.0600864, -0.6532185, 0.6202452
1: -0.2812218, 0.2745463, -0.2588896, 0.2532025, -0.5344243, 0.5334358
2: -0.1969118, 0.3704652, -0.1771662, 0.3451462, -0.5420580, 0.5476314
3: -0.2047744, 0.2773086, -0.1885284, 0.2572496, -0.4620240, 0.4658371
4: -0.2944947, 0.2555920, -0.2738361, 0.2322560, -0.5267507, 0.5294281
5: -0.3162411, 0.4140835, -0.2931452, 0.3841451, -0.7003862, 0.7072287
6: -0.2230460, 0.3055303, -0.2035349, 0.2823564, -0.5054023, 0.5090652
7: -0.3068058, 0.3130229, -0.2857333, 0.2886537, -0.5954595, 0.5987562
8: -0.2935277, 0.3808356, -0.2716854, 0.3521765, -0.6457043, 0.6525210
9: -0.2858101, 0.3680643, -0.2631110, 0.3390844, -0.6248945, 0.6311753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6213716
time: 1.20 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6213716
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.4594503, 1.0581031, 0.5188014, 1.0498079, -0.5903576, 0.5393017
1: -0.2505482, 0.2452306, -0.2159342, 0.2118887, -0.4624369, 0.4611647
2: -0.1697910, 0.3356894, -0.1393901, 0.2961376, -0.4659286, 0.4750796
3: -0.1824603, 0.2497573, -0.1571518, 0.2184221, -0.4008824, 0.4069092
4: -0.2661199, 0.2235395, -0.2338485, 0.1876050, -0.4537250, 0.4573880
5: -0.2845187, 0.3729630, -0.2484399, 0.3269465, -0.6114652, 0.6214030
6: -0.1962472, 0.2737007, -0.1660696, 0.2374996, -0.4337468, 0.4397703
7: -0.2778627, 0.2795514, -0.2452726, 0.2414834, -0.5193461, 0.5248240
8: -0.2635272, 0.3414720, -0.2294240, 0.2967027, -0.5602298, 0.5708960
9: -0.2546327, 0.3282602, -0.2191736, 0.2836619, -0.5382946, 0.5474338

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213409, upper bound: 0.6202619
time: 1.19 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6202619, upper bound: 0.6202619
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.3817502, 1.0688803, 0.4751050, 1.0559318, -0.6741816, 0.5937753
1: -0.2958741, 0.2885502, -0.2414161, 0.2365026, -0.5323768, 0.5299663
2: -0.2098669, 0.3870770, -0.1617167, 0.3253360, -0.5352029, 0.5487937
3: -0.2154335, 0.2904696, -0.1758170, 0.2415547, -0.4569882, 0.4662866
4: -0.3080488, 0.2709029, -0.2576723, 0.2139970, -0.5220459, 0.5285752
5: -0.3313941, 0.4337258, -0.2750744, 0.3607206, -0.6921147, 0.7088002
6: -0.2358473, 0.3207349, -0.1882688, 0.2642243, -0.5000716, 0.5090036
7: -0.3206315, 0.3290115, -0.2692457, 0.2695864, -0.5902179, 0.5982572
8: -0.3078586, 0.3996390, -0.2545954, 0.3297527, -0.6376113, 0.6542343
9: -0.3007030, 0.3870779, -0.2453506, 0.3164099, -0.6171130, 0.6324284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 111

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6249942
time: 1.25 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6249942
time: 1.22 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.4420684, 1.0605139, 0.5356331, 1.0472660, -0.6051975, 0.5248808
1: -0.2606879, 0.2549213, -0.2061566, 0.2016499, -0.4623379, 0.4610778
2: -0.1787563, 0.3471851, -0.1317810, 0.2846219, -0.4633782, 0.4789661
3: -0.1898366, 0.2588649, -0.1493887, 0.2098421, -0.3996786, 0.4082537
4: -0.2754997, 0.2341351, -0.2242022, 0.1785855, -0.4540852, 0.4583373
5: -0.2950051, 0.3865560, -0.2368979, 0.3162018, -0.6112068, 0.6234539
6: -0.2051061, 0.2842224, -0.1589141, 0.2259185, -0.4310246, 0.4431365
7: -0.2874303, 0.2906159, -0.2364091, 0.2299363, -0.5173665, 0.5270250
8: -0.2734444, 0.3544844, -0.2192159, 0.2823803, -0.5558247, 0.5737003
9: -0.2649389, 0.3414181, -0.2086956, 0.2725946, -0.5375335, 0.5501137

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 111

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6227149
time: 1.15 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6227149
time: 1.15 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.3315679, 1.0758410, 0.4749137, 1.0559582, -0.7243903, 0.6009272
1: -0.3251477, 0.3165280, -0.2415276, 0.2366092, -0.5617569, 0.5580555
2: -0.2357497, 0.4202657, -0.1618152, 0.3254624, -0.5612121, 0.5820810
3: -0.2367291, 0.3167636, -0.1758981, 0.2416549, -0.4783840, 0.4926617
4: -0.3351285, 0.3014925, -0.2577753, 0.2141135, -0.5492420, 0.5592679
5: -0.3616687, 0.4729692, -0.2751897, 0.3608701, -0.7225388, 0.7481589
6: -0.2614229, 0.3511117, -0.1883661, 0.2643400, -0.5257630, 0.5394778
7: -0.3482537, 0.3609552, -0.2693509, 0.2697081, -0.6179618, 0.6303061
8: -0.3364900, 0.4372058, -0.2547045, 0.3298959, -0.6663858, 0.6919103
9: -0.3304576, 0.4250651, -0.2454639, 0.3165545, -0.6470121, 0.6705290

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 111

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213626, upper bound: 0.6249899
time: 1.17 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213626, upper bound: 0.6249899
time: 1.23 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.3877308, 1.0680510, 0.5355136, 1.0472841, -0.6595533, 0.5325373
1: -0.2923854, 0.2852157, -0.2062259, 0.2017214, -0.4941068, 0.4914416
2: -0.2067822, 0.3831216, -0.1318350, 0.2847011, -0.4914833, 0.5149565
3: -0.2128955, 0.2873359, -0.1494438, 0.2098997, -0.4227952, 0.4367797
4: -0.3048215, 0.2672573, -0.2242692, 0.1786494, -0.4834709, 0.4915265
5: -0.3277861, 0.4290489, -0.2369797, 0.3162771, -0.6440632, 0.6660286
6: -0.2327992, 0.3171145, -0.1589626, 0.2260008, -0.4588000, 0.4760771
7: -0.3173395, 0.3252044, -0.2364720, 0.2300168, -0.5473562, 0.5616764
8: -0.3044463, 0.3951617, -0.2192869, 0.2824820, -0.5869283, 0.6144486
9: -0.2971568, 0.3825506, -0.2087681, 0.2726731, -0.5698299, 0.5913186

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 111

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213626, upper bound: 0.6227146
time: 1.13 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213626, upper bound: 0.6227146
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.4062446, 1.0654830, 0.4390577, 1.0609317, -0.6546871, 0.6264253
1: -0.2815855, 0.2748939, -0.2624441, 0.2566000, -0.5381854, 0.5373380
2: -0.1972332, 0.3708774, -0.1803091, 0.3491762, -0.5464094, 0.5511866
3: -0.2050389, 0.2776354, -0.1911143, 0.2604425, -0.4654814, 0.4687497
4: -0.2948310, 0.2559720, -0.2771243, 0.2359704, -0.5308014, 0.5330963
5: -0.3166170, 0.4145707, -0.2968214, 0.3889105, -0.7055275, 0.7113920
6: -0.2233637, 0.3059075, -0.2066405, 0.2860449, -0.5094086, 0.5125480
7: -0.3071489, 0.3134195, -0.2890876, 0.2925324, -0.5996813, 0.6025071
8: -0.2938834, 0.3813021, -0.2751621, 0.3567381, -0.6506215, 0.6564642
9: -0.2861796, 0.3685359, -0.2667241, 0.3436973, -0.6298770, 0.6352600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6213716
time: 1.16 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6213716
time: 1.19 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.4675888, 1.0569741, 0.5122252, 1.0507827, -0.5831939, 0.5447489
1: -0.2458007, 0.2406932, -0.2197622, 0.2158072, -0.4616079, 0.4604554
2: -0.1655935, 0.3303070, -0.1425710, 0.3007862, -0.4663796, 0.4728780
3: -0.1790067, 0.2454930, -0.1600646, 0.2221049, -0.4011116, 0.4055576
4: -0.2617282, 0.2185788, -0.2376412, 0.1913697, -0.4530979, 0.4562201
5: -0.2796088, 0.3665987, -0.2526802, 0.3316920, -0.6113009, 0.6192789
6: -0.1920995, 0.2687742, -0.1693503, 0.2417542, -0.4338537, 0.4381245
7: -0.2733830, 0.2743710, -0.2488135, 0.2459576, -0.5193406, 0.5231845
8: -0.2588838, 0.3353796, -0.2334166, 0.3019642, -0.5608480, 0.5687963
9: -0.2498072, 0.3220997, -0.2233410, 0.2883106, -0.5381178, 0.5454407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6218435, upper bound: 0.6202619
time: 1.19 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6206941, upper bound: 0.6202619
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.3625222, 1.0715474, 0.4386464, 1.0609887, -0.6984665, 0.6329010
1: -0.3070907, 0.2992702, -0.2626840, 0.2568291, -0.5639198, 0.5619542
2: -0.2197843, 0.3997938, -0.1805212, 0.3494482, -0.5692325, 0.5803150
3: -0.2235932, 0.3005445, -0.1912888, 0.2606578, -0.4842510, 0.4918332
4: -0.3184249, 0.2826237, -0.2773462, 0.2362209, -0.5546458, 0.5599699
5: -0.3429943, 0.4487625, -0.2970695, 0.3892320, -0.7322263, 0.7458320
6: -0.2456470, 0.3323743, -0.2068500, 0.2862938, -0.5319408, 0.5392243
7: -0.3312154, 0.3412511, -0.2893138, 0.2927942, -0.6240095, 0.6305649
8: -0.3188291, 0.4140332, -0.2753967, 0.3570459, -0.6758751, 0.6894299
9: -0.3121038, 0.4016332, -0.2669678, 0.3440085, -0.6561123, 0.6686010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6213716
time: 1.07 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6213716
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.4198285, 1.0635986, 0.5118777, 1.0508311, -0.6310025, 0.5517210
1: -0.2736613, 0.2673205, -0.2199650, 0.2160010, -0.4896623, 0.4872855
2: -0.1902270, 0.3618937, -0.1427503, 0.3010159, -0.4912430, 0.5046439
3: -0.1992744, 0.2705178, -0.1602120, 0.2222869, -0.4215613, 0.4307298
4: -0.2875008, 0.2476918, -0.2378288, 0.1915816, -0.4790824, 0.4855206
5: -0.3084221, 0.4039480, -0.2528899, 0.3319637, -0.6403858, 0.6568379
6: -0.2164406, 0.2976848, -0.1695274, 0.2419646, -0.4584053, 0.4672123
7: -0.2996718, 0.3047727, -0.2490047, 0.2461789, -0.5458506, 0.5537774
8: -0.2861332, 0.3711333, -0.2336150, 0.3022243, -0.5883576, 0.6047483
9: -0.2781254, 0.3582533, -0.2235471, 0.2885736, -0.5666990, 0.5818004

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6200992, upper bound: 0.6210758
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6213716
time: 1.09 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.87 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6246871
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6246871
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6226884
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6226884
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 0, lower bound: -0.6213626, upper bound: 0.6246661
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 0, lower bound: -0.6213626, upper bound: 0.6246661
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 0, lower bound: -0.6213626, upper bound: 0.6226865
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 0, lower bound: -0.6213626, upper bound: 0.6226865
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6213716
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6213716
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 0, lower bound: -0.6218435, upper bound: 0.6202619
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 0, lower bound: -0.6206941, upper bound: 0.6202619
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6213716
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6213716
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 0, lower bound: -0.6213409, upper bound: 0.6202619
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 0, lower bound: -0.6202619, upper bound: 0.6202619
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6249942
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6249942
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6227149
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6227149
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 0, lower bound: -0.6213626, upper bound: 0.6249899
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 0, lower bound: -0.6213626, upper bound: 0.6249899
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 0, lower bound: -0.6213626, upper bound: 0.6227146
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 0, lower bound: -0.6213626, upper bound: 0.6227146
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6213716
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6213716
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 0, lower bound: -0.6218435, upper bound: 0.6202619
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 0, lower bound: -0.6206941, upper bound: 0.6202619
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6213716
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6213716
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 0, lower bound: -0.6200992, upper bound: 0.6210758
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6213716

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.5021594, 1.0521791, 0.4824845, 1.0549082, -0.5527488, 0.5696945
1: -0.2256341, 0.2214193, -0.2371112, 0.2323883, -0.4580224, 0.4585305
2: -0.1477628, 0.3074434, -0.1579106, 0.3204553, -0.4682181, 0.4653539
3: -0.1643362, 0.2273792, -0.1726854, 0.2376881, -0.4020243, 0.4000646
4: -0.2430730, 0.1975056, -0.2536900, 0.2094986, -0.4525716, 0.4511956
5: -0.2587529, 0.3395638, -0.2706224, 0.3549497, -0.6137027, 0.6101862
6: -0.1744805, 0.2478475, -0.1845077, 0.2597573, -0.4342378, 0.4323553
7: -0.2543540, 0.2523651, -0.2651837, 0.2648890, -0.5192430, 0.5175488
8: -0.2391597, 0.3094997, -0.2503850, 0.3242283, -0.5633880, 0.5598848
9: -0.2293095, 0.2959302, -0.2409750, 0.3108238, -0.5401333, 0.5369053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6200984
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6213442
time: 1.28 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.4700723, 1.0566295, 0.4824845, 1.0549082, -0.5848359, 0.5741450
1: -0.2443519, 0.2393085, -0.2371112, 0.2323883, -0.4767402, 0.4764197
2: -0.1643125, 0.3286644, -0.1579106, 0.3204553, -0.4847679, 0.4865749
3: -0.1779528, 0.2441918, -0.1726854, 0.2376881, -0.4156409, 0.4168772
4: -0.2603881, 0.2170648, -0.2536900, 0.2094986, -0.4698867, 0.4707548
5: -0.2781106, 0.3646565, -0.2706224, 0.3549497, -0.6330603, 0.6352788
6: -0.1908337, 0.2672708, -0.1845077, 0.2597573, -0.4505910, 0.4517785
7: -0.2720159, 0.2727900, -0.2651837, 0.2648890, -0.5369049, 0.5379738
8: -0.2574668, 0.3335203, -0.2503850, 0.3242283, -0.5816951, 0.5839053
9: -0.2483348, 0.3202196, -0.2409750, 0.3108238, -0.5591586, 0.5611947

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=42, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6200984
time: 1.25 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6213442
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.5502698, 1.0450853, 0.5427965, 1.0461988, -0.4959291, 0.5022888
1: -0.1976541, 0.1928594, -0.2019952, 0.1973477, -0.3950018, 0.3948547
2: -0.1251641, 0.2749091, -0.1285425, 0.2798682, -0.4050323, 0.4034516
3: -0.1426380, 0.2027633, -0.1460848, 0.2063776, -0.3490156, 0.3488482
4: -0.2159936, 0.1707420, -0.2201848, 0.1747467, -0.3907403, 0.3909267
5: -0.2268611, 0.3069639, -0.2319856, 0.3116806, -0.5385417, 0.5389495
6: -0.1529734, 0.2158478, -0.1560066, 0.2209897, -0.3739631, 0.3718545
7: -0.2287014, 0.2200619, -0.2326368, 0.2251036, -0.4538051, 0.4526988
8: -0.2105025, 0.2699260, -0.2149514, 0.2762848, -0.4867873, 0.4848774
9: -0.1998134, 0.2629705, -0.2043484, 0.2678843, -0.4676978, 0.4673190

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6206543, upper bound: 0.6220167
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6206543, upper bound: 0.6212123
time: 1.28 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.5246415, 1.0489039, 0.5427965, 1.0461988, -0.5215573, 0.5061074
1: -0.2125416, 0.2082512, -0.2019952, 0.1973477, -0.4098893, 0.4102464
2: -0.1367501, 0.2919158, -0.1285425, 0.2798682, -0.4166183, 0.4204584
3: -0.1544584, 0.2151578, -0.1460848, 0.2063776, -0.3608360, 0.3612427
4: -0.2303668, 0.1844756, -0.2201848, 0.1747467, -0.4051135, 0.4046604
5: -0.2444352, 0.3231390, -0.2319856, 0.3116806, -0.5561158, 0.5551246
6: -0.1633756, 0.2334814, -0.1560066, 0.2209897, -0.3843653, 0.3894880
7: -0.2421973, 0.2373514, -0.2326368, 0.2251036, -0.4673010, 0.4699883
8: -0.2257592, 0.2917333, -0.2149514, 0.2762848, -0.5020441, 0.5066848
9: -0.2153659, 0.2798220, -0.2043484, 0.2678843, -0.4832502, 0.4841704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=42, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6206543, upper bound: 0.6220167
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6206543, upper bound: 0.6212123
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.4464990, 1.0598994, 0.4823077, 1.0549325, -0.6084334, 0.5775917
1: -0.2581033, 0.2524512, -0.2372145, 0.2324869, -0.4905903, 0.4896657
2: -0.1764710, 0.3442549, -0.1580018, 0.3205723, -0.4970434, 0.5022566
3: -0.1879565, 0.2565434, -0.1727605, 0.2377808, -0.4257372, 0.4293039
4: -0.2731088, 0.2314344, -0.2537854, 0.2096065, -0.4827152, 0.4852198
5: -0.2923321, 0.3830912, -0.2707291, 0.3550880, -0.6474200, 0.6538203
6: -0.2028480, 0.2815405, -0.1845980, 0.2598642, -0.4627122, 0.4661385
7: -0.2849916, 0.2877955, -0.2652811, 0.2650016, -0.5499932, 0.5530766
8: -0.2709165, 0.3511675, -0.2504859, 0.3243607, -0.5952772, 0.6016533
9: -0.2623119, 0.3380641, -0.2410798, 0.3109577, -0.5732695, 0.5791440

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6181348, upper bound: 0.6200550
time: 1.21 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6181348, upper bound: 0.6213309
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.4229718, 1.0631627, 0.4823077, 1.0549325, -0.6319606, 0.5808550
1: -0.2718277, 0.2655680, -0.2372145, 0.2324869, -0.5043146, 0.5027825
2: -0.1886057, 0.3598147, -0.1580018, 0.3205723, -0.5091780, 0.5178164
3: -0.1979404, 0.2688708, -0.1727605, 0.2377808, -0.4357212, 0.4416313
4: -0.2858045, 0.2457756, -0.2537854, 0.2096065, -0.4954110, 0.4995610
5: -0.3065256, 0.4014897, -0.2707291, 0.3550880, -0.6616136, 0.6722188
6: -0.2148385, 0.2957820, -0.1845980, 0.2598642, -0.4747027, 0.4803799
7: -0.2979415, 0.3027717, -0.2652811, 0.2650016, -0.5629431, 0.5680528
8: -0.2843397, 0.3687799, -0.2504859, 0.3243607, -0.6087004, 0.6192658
9: -0.2762616, 0.3558736, -0.2410798, 0.3109577, -0.5872192, 0.5969535

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=42, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6181348, upper bound: 0.6200550
time: 1.27 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6181348, upper bound: 0.6213309
time: 1.23 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.4991377, 1.0525982, 0.5426989, 1.0462135, -0.5470758, 0.5098994
1: -0.2273966, 0.2231037, -0.2020521, 0.1974065, -0.4248031, 0.4251558
2: -0.1493211, 0.3094416, -0.1285868, 0.2799332, -0.4292544, 0.4380283
3: -0.1656183, 0.2289622, -0.1461299, 0.2064249, -0.3720432, 0.3750921
4: -0.2447034, 0.1993475, -0.2202396, 0.1747991, -0.4195025, 0.4195871
5: -0.2605757, 0.3419266, -0.2320528, 0.3117423, -0.5723181, 0.5739794
6: -0.1760204, 0.2496763, -0.1560463, 0.2210570, -0.3970774, 0.4057227
7: -0.2560172, 0.2542883, -0.2326884, 0.2251695, -0.4811867, 0.4869766
8: -0.2408835, 0.3117616, -0.2150097, 0.2763681, -0.5172516, 0.5267712
9: -0.2311009, 0.2982175, -0.2044079, 0.2679488, -0.4990497, 0.5026253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6201944, upper bound: 0.6220167
time: 3.43 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6201944, upper bound: 0.6212118
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.4763891, 1.0557536, 0.5426989, 1.0462135, -0.5698244, 0.5130547
1: -0.2406669, 0.2357867, -0.2020521, 0.1974065, -0.4380734, 0.4378388
2: -0.1610544, 0.3244866, -0.1285868, 0.2799332, -0.4409876, 0.4530734
3: -0.1752721, 0.2408818, -0.1461299, 0.2064249, -0.3816970, 0.3870117
4: -0.2569793, 0.2132141, -0.2202396, 0.1747991, -0.4317784, 0.4334537
5: -0.2742996, 0.3597166, -0.2320528, 0.3117423, -0.5860419, 0.5917693
6: -0.1876143, 0.2634469, -0.1560463, 0.2210570, -0.4086713, 0.4194932
7: -0.2685388, 0.2687691, -0.2326884, 0.2251695, -0.4937083, 0.5014575
8: -0.2538627, 0.3287914, -0.2150097, 0.2763681, -0.5302308, 0.5438011
9: -0.2445892, 0.3154378, -0.2044079, 0.2679488, -0.5125380, 0.5198457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=42, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6201944, upper bound: 0.6220167
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6201944, upper bound: 0.6212118
time: 1.23 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.5025843, 1.0521200, 0.4455253, 1.0600346, -0.5574503, 0.6065947
1: -0.2253862, 0.2211823, -0.2586713, 0.2529940, -0.4783803, 0.4798537
2: -0.1475435, 0.3071623, -0.1769733, 0.3448989, -0.4924424, 0.4841356
3: -0.1641558, 0.2271564, -0.1883696, 0.2570534, -0.4212092, 0.4155259
4: -0.2428437, 0.1972465, -0.2736342, 0.2320278, -0.4748716, 0.4708807
5: -0.2584964, 0.3392314, -0.2929194, 0.3838527, -0.6423490, 0.6321509
6: -0.1742638, 0.2475902, -0.2033442, 0.2821299, -0.4563938, 0.4509344
7: -0.2541201, 0.2520945, -0.2855275, 0.2884154, -0.5425355, 0.5376220
8: -0.2389172, 0.3091814, -0.2714720, 0.3518966, -0.5908138, 0.5806534
9: -0.2290573, 0.2956086, -0.2628891, 0.3388011, -0.5678585, 0.5584978

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6203421
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6216087
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.5504214, 1.0450628, 0.4455253, 1.0600346, -0.5096133, 0.5995375
1: -0.1975661, 0.1927686, -0.2586713, 0.2529940, -0.4505602, 0.4514400
2: -0.1250957, 0.2748086, -0.1769733, 0.3448989, -0.4699945, 0.4517819
3: -0.1425681, 0.2026902, -0.1883696, 0.2570534, -0.3996215, 0.3910598
4: -0.2159086, 0.1706609, -0.2736342, 0.2320278, -0.4479365, 0.4442951
5: -0.2267573, 0.3068684, -0.2929194, 0.3838527, -0.6106099, 0.5997878
6: -0.1529120, 0.2157436, -0.2033442, 0.2821299, -0.4350419, 0.4190877
7: -0.2286218, 0.2199599, -0.2855275, 0.2884154, -0.5170372, 0.5054874
8: -0.2104124, 0.2697969, -0.2714720, 0.3518966, -0.5623090, 0.5412689
9: -0.1997216, 0.2628711, -0.2628891, 0.3388011, -0.5385227, 0.5257602

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=42, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6203421
time: 1.34 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6216087
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.5256526, 1.0487531, 0.5234109, 1.0490872, -0.5234345, 0.5253422
1: -0.2119544, 0.2076437, -0.2132564, 0.2089901, -0.4209445, 0.4209001
2: -0.1362929, 0.2912450, -0.1373063, 0.2927324, -0.4290253, 0.4285513
3: -0.1539919, 0.2146688, -0.1550258, 0.2157529, -0.3697448, 0.3696946
4: -0.2297996, 0.1839338, -0.2310569, 0.1851350, -0.4149346, 0.4149907
5: -0.2437418, 0.3225008, -0.2452790, 0.3239158, -0.5676576, 0.5677798
6: -0.1629651, 0.2327856, -0.1638749, 0.2343280, -0.3972931, 0.3966605
7: -0.2416648, 0.2366693, -0.2428453, 0.2381815, -0.4798464, 0.4795146
8: -0.2251574, 0.2908728, -0.2264917, 0.2927803, -0.5179377, 0.5173645
9: -0.2147523, 0.2791571, -0.2161126, 0.2806309, -0.4953831, 0.4952697

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6206278, upper bound: 0.6187256
time: 1.23 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6218435, upper bound: 0.6202619
time: 1.37 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.4665194, 1.0571226, 0.5276337, 1.0484580, -0.5819386, 0.5294889
1: -0.2464246, 0.2412892, -0.2108034, 0.2064540, -0.4528786, 0.4520926
2: -0.1661450, 0.3310142, -0.1353973, 0.2899303, -0.4560753, 0.4664115
3: -0.1794605, 0.2460534, -0.1530782, 0.2137107, -0.3931712, 0.3991315
4: -0.2623053, 0.2192305, -0.2286885, 0.1828722, -0.4451775, 0.4479191
5: -0.2802540, 0.3674348, -0.2423833, 0.3212505, -0.6015046, 0.6098181
6: -0.1926445, 0.2694215, -0.1621610, 0.2314226, -0.4240671, 0.4315825
7: -0.2739715, 0.2750517, -0.2406216, 0.2353328, -0.5093043, 0.5156733
8: -0.2594940, 0.3361800, -0.2239779, 0.2891870, -0.5486810, 0.5601579
9: -0.2504412, 0.3229091, -0.2135500, 0.2778544, -0.5282956, 0.5364591

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6199196, upper bound: 0.6187256
time: 1.20 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6206941, upper bound: 0.6202619
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.4569337, 1.0584520, 0.4451512, 1.0600864, -0.6031527, 0.6133008
1: -0.2520162, 0.2466335, -0.2588896, 0.2532025, -0.5052187, 0.5055230
2: -0.1710890, 0.3373535, -0.1771662, 0.3451462, -0.5162352, 0.5145197
3: -0.1835283, 0.2510758, -0.1885284, 0.2572496, -0.4407779, 0.4396042
4: -0.2674778, 0.2250736, -0.2738361, 0.2322560, -0.4997337, 0.4989097
5: -0.2860368, 0.3749309, -0.2931452, 0.3841451, -0.6701820, 0.6680762
6: -0.1975298, 0.2752238, -0.2035349, 0.2823564, -0.4798862, 0.4787587
7: -0.2792478, 0.2811534, -0.2857333, 0.2886537, -0.5679014, 0.5668867
8: -0.2649629, 0.3433558, -0.2716854, 0.3521765, -0.6171393, 0.6150413
9: -0.2561247, 0.3301651, -0.2631110, 0.3390844, -0.5952092, 0.5932761

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6181361, upper bound: 0.6203180
time: 1.19 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6181361, upper bound: 0.6215980
time: 1.13 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.5019262, 1.0522114, 0.4451512, 1.0600864, -0.5581601, 0.6070602
1: -0.2257701, 0.2215492, -0.2588896, 0.2532025, -0.4789726, 0.4804388
2: -0.1478831, 0.3075975, -0.1771662, 0.3451462, -0.4930293, 0.4847637
3: -0.1644351, 0.2275013, -0.1885284, 0.2572496, -0.4216847, 0.4160297
4: -0.2431988, 0.1976478, -0.2738361, 0.2322560, -0.4754548, 0.4714839
5: -0.2588935, 0.3397461, -0.2931452, 0.3841451, -0.6430387, 0.6328914
6: -0.1745993, 0.2479886, -0.2035349, 0.2823564, -0.4569557, 0.4515234
7: -0.2544825, 0.2525135, -0.2857333, 0.2886537, -0.5431362, 0.5382468
8: -0.2392927, 0.3096742, -0.2716854, 0.3521765, -0.5914692, 0.5813596
9: -0.2294477, 0.2961070, -0.2631110, 0.3390844, -0.5685321, 0.5592179

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=42, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6181361, upper bound: 0.6203180
time: 1.18 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6181361, upper bound: 0.6215980
time: 3.55 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.4769465, 1.0556761, 0.5231144, 1.0491313, -0.5721848, 0.5325617
1: -0.2403418, 0.2354760, -0.2134287, 0.2091682, -0.4495100, 0.4489046
2: -0.1607669, 0.3241179, -0.1374404, 0.2929291, -0.4536961, 0.4615583
3: -0.1750356, 0.2405898, -0.1551626, 0.2158964, -0.3909320, 0.3957523
4: -0.2566783, 0.2128745, -0.2312230, 0.1852940, -0.4419723, 0.4440975
5: -0.2739634, 0.3592806, -0.2454824, 0.3241027, -0.5980661, 0.6047629
6: -0.1873302, 0.2631095, -0.1639953, 0.2345320, -0.4218622, 0.4271048
7: -0.2682320, 0.2684143, -0.2430014, 0.2383815, -0.5066136, 0.5114156
8: -0.2535447, 0.3283741, -0.2266682, 0.2930326, -0.5465773, 0.5550423
9: -0.2442586, 0.3150159, -0.2162925, 0.2808260, -0.5250846, 0.5313085

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6205137, upper bound: 0.6187256
time: 1.22 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213409, upper bound: 0.6202619
time: 1.22 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.4213130, 1.0633929, 0.5272436, 1.0485162, -0.6272031, 0.5361493
1: -0.2727953, 0.2664929, -0.2110300, 0.2066884, -0.4794837, 0.4775229
2: -0.1894613, 0.3609117, -0.1355736, 0.2901890, -0.4796504, 0.4964854
3: -0.1986444, 0.2697399, -0.1532581, 0.2138993, -0.4125437, 0.4229981
4: -0.2866997, 0.2467868, -0.2289073, 0.1830813, -0.4697809, 0.4756941
5: -0.3075264, 0.4027870, -0.2426509, 0.3214967, -0.6290231, 0.6454380
6: -0.2156840, 0.2967862, -0.1623193, 0.2316909, -0.4473749, 0.4591056
7: -0.2988546, 0.3038277, -0.2408270, 0.2355959, -0.5344505, 0.5446547
8: -0.2852862, 0.3700218, -0.2242101, 0.2895189, -0.5748051, 0.5942320
9: -0.2772452, 0.3571294, -0.2137866, 0.2781109, -0.5553561, 0.5709160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6197221, upper bound: 0.6187256
time: 1.19 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6202619, upper bound: 0.6202619
time: 1.09 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.4588694, 1.0581836, 0.4751050, 1.0559318, -0.5970624, 0.5830786
1: -0.2508871, 0.2455544, -0.2414161, 0.2365026, -0.4873897, 0.4869705
2: -0.1700907, 0.3360735, -0.1617167, 0.3253360, -0.4954267, 0.4977902
3: -0.1827069, 0.2500617, -0.1758170, 0.2415547, -0.4242616, 0.4258786
4: -0.2664334, 0.2238937, -0.2576723, 0.2139970, -0.4804304, 0.4815660
5: -0.2848692, 0.3734171, -0.2750744, 0.3607206, -0.6455898, 0.6484916
6: -0.1965434, 0.2740521, -0.1882688, 0.2642243, -0.4607677, 0.4623209
7: -0.2781824, 0.2799212, -0.2692457, 0.2695864, -0.5477688, 0.5491670
8: -0.2638585, 0.3419070, -0.2545954, 0.3297527, -0.5936112, 0.5965024
9: -0.2549771, 0.3287001, -0.2453506, 0.3164099, -0.5713871, 0.5740507

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6249942
time: 1.29 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6249942
time: 1.29 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.4234510, 1.0630963, 0.4751050, 1.0559318, -0.6324809, 0.5879913
1: -0.2715483, 0.2653010, -0.2414161, 0.2365026, -0.5080509, 0.5067171
2: -0.1883587, 0.3594979, -0.1617167, 0.3253360, -0.5136947, 0.5212146
3: -0.1977373, 0.2686198, -0.1758170, 0.2415547, -0.4392920, 0.4444368
4: -0.2855461, 0.2454836, -0.2576723, 0.2139970, -0.4995431, 0.5031559
5: -0.3062367, 0.4011153, -0.2750744, 0.3607206, -0.6669573, 0.6761897
6: -0.2145945, 0.2954921, -0.1882688, 0.2642243, -0.4788188, 0.4837609
7: -0.2976781, 0.3024670, -0.2692457, 0.2695864, -0.5672644, 0.5717128
8: -0.2840665, 0.3684217, -0.2545954, 0.3297527, -0.6138192, 0.6230171
9: -0.2759777, 0.3555113, -0.2453506, 0.3164099, -0.5923876, 0.6008618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=42, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6249942
time: 1.19 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6249942
time: 1.19 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.5177581, 1.0499723, 0.5356331, 1.0472660, -0.5295079, 0.5143392
1: -0.2165401, 0.2125497, -0.2061566, 0.2016499, -0.4181900, 0.4187062
2: -0.1398617, 0.2969218, -0.1317810, 0.2846219, -0.4244835, 0.4287028
3: -0.1576329, 0.2190433, -0.1493887, 0.2098421, -0.3674750, 0.3684320
4: -0.2344882, 0.1881642, -0.2242022, 0.1785855, -0.4130737, 0.4123664
5: -0.2491552, 0.3276372, -0.2368979, 0.3162018, -0.5653570, 0.5645351
6: -0.1665789, 0.2382173, -0.1589141, 0.2259185, -0.3924974, 0.3971315
7: -0.2458218, 0.2422382, -0.2364091, 0.2299363, -0.4757581, 0.4786473
8: -0.2300950, 0.2975902, -0.2192159, 0.2823803, -0.5124753, 0.5168061
9: -0.2198766, 0.2843478, -0.2086956, 0.2725946, -0.4924712, 0.4930434

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6227149
time: 1.12 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6227149
time: 1.15 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.4860784, 1.0544096, 0.5356331, 1.0472660, -0.5611876, 0.5187765
1: -0.2350148, 0.2303847, -0.2061566, 0.2016499, -0.4366648, 0.4365413
2: -0.1560569, 0.3180788, -0.1317810, 0.2846219, -0.4406788, 0.4498597
3: -0.1711604, 0.2358051, -0.1493887, 0.2098421, -0.3810024, 0.3851938
4: -0.2517508, 0.2073080, -0.2242022, 0.1785855, -0.4303363, 0.4315102
5: -0.2684543, 0.3521394, -0.2368979, 0.3162018, -0.5846561, 0.5890373
6: -0.1826762, 0.2575817, -0.1589141, 0.2259185, -0.4085947, 0.4164958
7: -0.2632056, 0.2626015, -0.2364091, 0.2299363, -0.4931418, 0.4990106
8: -0.2483346, 0.3215379, -0.2192159, 0.2823803, -0.5307150, 0.5407538
9: -0.2388442, 0.3081034, -0.2086956, 0.2725946, -0.5114388, 0.5167990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=42, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6227149
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6227149
time: 1.15 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.4034982, 1.0658638, 0.4749137, 1.0559582, -0.6524599, 0.5909501
1: -0.2831876, 0.2764251, -0.2415276, 0.2366092, -0.5197968, 0.5179527
2: -0.1986498, 0.3726937, -0.1618152, 0.3254624, -0.5241122, 0.5345089
3: -0.2062044, 0.2790744, -0.1758981, 0.2416549, -0.4478593, 0.4549725
4: -0.2963131, 0.2576461, -0.2577753, 0.2141135, -0.5104266, 0.5154215
5: -0.3182740, 0.4167187, -0.2751897, 0.3608701, -0.6791441, 0.6919084
6: -0.2247634, 0.3075702, -0.1883661, 0.2643400, -0.4891034, 0.4959363
7: -0.3086607, 0.3151678, -0.2693509, 0.2697081, -0.5783688, 0.5845187
8: -0.2954503, 0.3833583, -0.2547045, 0.3298959, -0.6253461, 0.6380628
9: -0.2878081, 0.3706150, -0.2454639, 0.3165545, -0.6043626, 0.6160789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213626, upper bound: 0.6249899
time: 1.36 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213626, upper bound: 0.6249899
time: 1.12 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.3803581, 1.0690734, 0.4749137, 1.0559582, -0.6756001, 0.5941597
1: -0.2966862, 0.2893263, -0.2415276, 0.2366092, -0.5332954, 0.5308539
2: -0.2105850, 0.3879978, -0.1618152, 0.3254624, -0.5360474, 0.5498131
3: -0.2160244, 0.2911992, -0.1758981, 0.2416549, -0.4576793, 0.4670973
4: -0.3088002, 0.2717518, -0.2577753, 0.2141135, -0.5229137, 0.5295271
5: -0.3322342, 0.4348146, -0.2751897, 0.3608701, -0.6931043, 0.7100043
6: -0.2365569, 0.3215777, -0.1883661, 0.2643400, -0.5008969, 0.5099438
7: -0.3213978, 0.3298978, -0.2693509, 0.2697081, -0.5911059, 0.5992487
8: -0.3086529, 0.4006813, -0.2547045, 0.3298959, -0.6385488, 0.6553857
9: -0.3015285, 0.3881317, -0.2454639, 0.3165545, -0.6180831, 0.6335956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=42, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213626, upper bound: 0.6249899
time: 1.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213626, upper bound: 0.6249899
time: 1.33 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.4605448, 1.0579513, 0.5355136, 1.0472841, -0.5867393, 0.5224377
1: -0.2499096, 0.2446203, -0.2062259, 0.2017214, -0.4516310, 0.4508462
2: -0.1692264, 0.3349654, -0.1318350, 0.2847011, -0.4539275, 0.4668004
3: -0.1819959, 0.2491838, -0.1494438, 0.2098997, -0.3918956, 0.3986275
4: -0.2655292, 0.2228723, -0.2242692, 0.1786494, -0.4441786, 0.4471415
5: -0.2838584, 0.3721070, -0.2369797, 0.3162771, -0.6001356, 0.6090868
6: -0.1956894, 0.2730380, -0.1589626, 0.2260008, -0.4216902, 0.4320006
7: -0.2772601, 0.2788546, -0.2364720, 0.2300168, -0.5072769, 0.5153265
8: -0.2629026, 0.3406527, -0.2192869, 0.2824820, -0.5453846, 0.5599396
9: -0.2539837, 0.3274317, -0.2087681, 0.2726731, -0.5266568, 0.5361997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213626, upper bound: 0.6227146
time: 1.46 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213626, upper bound: 0.6227146
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.4399798, 1.0608034, 0.5355136, 1.0472841, -0.6073043, 0.5252898
1: -0.2619062, 0.2560859, -0.2062259, 0.2017214, -0.4636276, 0.4623118
2: -0.1798335, 0.3485664, -0.1318350, 0.2847011, -0.4645346, 0.4804013
3: -0.1907230, 0.2599593, -0.1494438, 0.2098997, -0.4006227, 0.4094031
4: -0.2766268, 0.2354082, -0.2242692, 0.1786494, -0.4552761, 0.4596774
5: -0.2962650, 0.3881893, -0.2369797, 0.3162771, -0.6125422, 0.6251690
6: -0.2061705, 0.2854867, -0.1589626, 0.2260008, -0.4321713, 0.4444493
7: -0.2885799, 0.2919455, -0.2364720, 0.2300168, -0.5185967, 0.5284175
8: -0.2746360, 0.3560478, -0.2192869, 0.2824820, -0.5571181, 0.5753347
9: -0.2661773, 0.3429992, -0.2087681, 0.2726731, -0.5388504, 0.5517672

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=42, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213626, upper bound: 0.6227146
time: 1.20 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213626, upper bound: 0.6227146
time: 1.23 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.4493957, 1.0594977, 0.4390577, 1.0609317, -0.6115360, 0.6204400
1: -0.2564134, 0.2508360, -0.2624441, 0.2566000, -0.5130134, 0.5132801
2: -0.1749770, 0.3423389, -0.1803091, 0.3491762, -0.5241532, 0.5226481
3: -0.1867271, 0.2550255, -0.1911143, 0.2604425, -0.4471696, 0.4461398
4: -0.2715454, 0.2296685, -0.2771243, 0.2359704, -0.5075158, 0.5067928
5: -0.2905845, 0.3808258, -0.2968214, 0.3889105, -0.6794950, 0.6776472
6: -0.2013715, 0.2797869, -0.2066405, 0.2860449, -0.4874164, 0.4864274
7: -0.2833969, 0.2859517, -0.2890876, 0.2925324, -0.5759293, 0.5750393
8: -0.2692636, 0.3489989, -0.2751621, 0.3567381, -0.6260017, 0.6241610
9: -0.2605941, 0.3358713, -0.2667241, 0.3436973, -0.6042914, 0.6025954

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6253811
time: 1.16 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6253811
time: 1.15 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.5199930, 1.0496200, 0.4390577, 1.0609317, -0.5409387, 0.6105623
1: -0.2152419, 0.2111335, -0.2624441, 0.2566000, -0.4718419, 0.4735776
2: -0.1388515, 0.2952418, -0.1803091, 0.3491762, -0.4880277, 0.4755510
3: -0.1566022, 0.2177125, -0.1911143, 0.2604425, -0.4170447, 0.4088268
4: -0.2331175, 0.1869666, -0.2771243, 0.2359704, -0.4690879, 0.4640909
5: -0.2476228, 0.3261575, -0.2968214, 0.3889105, -0.6365333, 0.6229789
6: -0.1654878, 0.2366798, -0.2066405, 0.2860449, -0.4515328, 0.4433203
7: -0.2446451, 0.2406213, -0.2890876, 0.2925324, -0.5371775, 0.5297089
8: -0.2286576, 0.2956887, -0.2751621, 0.3567381, -0.5853957, 0.5708508
9: -0.2183705, 0.2828783, -0.2667241, 0.3436973, -0.5620677, 0.5496024

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=42, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6253811
time: 1.24 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6253811
time: 1.25 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.4840704, 1.0546880, 0.5169192, 1.0501046, -0.5660343, 0.5377688
1: -0.2361862, 0.2315043, -0.2170274, 0.2130813, -0.4492675, 0.4485317
2: -0.1570926, 0.3194066, -0.1402410, 0.2975526, -0.4546452, 0.4596475
3: -0.1720125, 0.2368571, -0.1580199, 0.2195430, -0.3915555, 0.3948770
4: -0.2528343, 0.2085319, -0.2350029, 0.1886137, -0.4414480, 0.4435349
5: -0.2696657, 0.3537096, -0.2497305, 0.3281927, -0.5978584, 0.6034400
6: -0.1836996, 0.2587972, -0.1669886, 0.2387946, -0.4224942, 0.4257858
7: -0.2643108, 0.2638797, -0.2462637, 0.2428454, -0.5071562, 0.5101434
8: -0.2494804, 0.3230411, -0.2306346, 0.2983040, -0.5477844, 0.5536757
9: -0.2400348, 0.3096234, -0.2204420, 0.2848994, -0.5249342, 0.5300654

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6217965, upper bound: 0.6202619
time: 1.16 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6217965, upper bound: 0.6202619
time: 1.18 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.4258893, 1.0627582, 0.5210741, 1.0494497, -0.6235604, 0.5416842
1: -0.2701258, 0.2639413, -0.2146139, 0.2104484, -0.4805743, 0.4785552
2: -0.1871010, 0.3578853, -0.1383627, 0.2944292, -0.4815302, 0.4962479
3: -0.1967024, 0.2673422, -0.1561036, 0.2170686, -0.4137710, 0.4234458
4: -0.2842302, 0.2439972, -0.2324544, 0.1863873, -0.4706175, 0.4764516
5: -0.3047656, 0.3992082, -0.2468814, 0.3254419, -0.6302075, 0.6460896
6: -0.2133518, 0.2940160, -0.1649600, 0.2359357, -0.4492875, 0.4589761
7: -0.2963358, 0.3009148, -0.2440758, 0.2398391, -0.5361749, 0.5449905
8: -0.2826752, 0.3665960, -0.2279623, 0.2947686, -0.5774438, 0.5945582
9: -0.2745318, 0.3536654, -0.2176419, 0.2821675, -0.5566993, 0.5713073

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6206941, upper bound: 0.6202619
time: 1.12 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6206941, upper bound: 0.6202619
time: 1.08 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.4072146, 1.0653484, 0.4386464, 1.0609887, -0.6537741, 0.6267020
1: -0.2810197, 0.2743532, -0.2626840, 0.2568291, -0.5378488, 0.5370372
2: -0.1967330, 0.3702359, -0.1805212, 0.3494482, -0.5461812, 0.5507571
3: -0.2046273, 0.2771271, -0.1912888, 0.2606578, -0.4652851, 0.4684158
4: -0.2943077, 0.2553808, -0.2773462, 0.2362209, -0.5305286, 0.5327270
5: -0.3160320, 0.4138122, -0.2970695, 0.3892320, -0.7052640, 0.7108816
6: -0.2228693, 0.3053206, -0.2068500, 0.2862938, -0.5091631, 0.5121706
7: -0.3066150, 0.3128022, -0.2893138, 0.2927942, -0.5994092, 0.6021160
8: -0.2933301, 0.3805762, -0.2753967, 0.3570459, -0.6503761, 0.6559729
9: -0.2856046, 0.3678018, -0.2669678, 0.3440085, -0.6296130, 0.6347696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6253733
time: 1.13 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6253734
time: 1.22 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.4697161, 1.0566792, 0.4386464, 1.0609887, -0.5912725, 0.6180329
1: -0.2445597, 0.2395070, -0.2626840, 0.2568291, -0.5013888, 0.5021910
2: -0.1644962, 0.3289000, -0.1805212, 0.3494482, -0.5139444, 0.5094212
3: -0.1781039, 0.2443784, -0.1912888, 0.2606578, -0.4387616, 0.4356672
4: -0.2605802, 0.2172819, -0.2773462, 0.2362209, -0.4968011, 0.4946281
5: -0.2783255, 0.3649350, -0.2970695, 0.3892320, -0.6675575, 0.6620044
6: -0.1910153, 0.2674863, -0.2068500, 0.2862938, -0.4773091, 0.4743363
7: -0.2722120, 0.2730168, -0.2893138, 0.2927942, -0.5650062, 0.5623306
8: -0.2576701, 0.3337868, -0.2753967, 0.3570459, -0.6147161, 0.6091834
9: -0.2485458, 0.3204893, -0.2669678, 0.3440085, -0.5925543, 0.5874571

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=42, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6253734
time: 1.22 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6253734
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.3895567, 1.0677977, 0.5232200, 1.0491157, -0.6595590, 0.5445777
1: -0.2913203, 0.2841977, -0.2133673, 0.2091046, -0.5004249, 0.4975650
2: -0.2058405, 0.3819142, -0.1373925, 0.2928590, -0.4986995, 0.5193067
3: -0.2121207, 0.2863794, -0.1551138, 0.2158452, -0.4279658, 0.4414932
4: -0.3038363, 0.2661445, -0.2311637, 0.1852372, -0.4890735, 0.4973082
5: -0.3266848, 0.4276211, -0.2454098, 0.3240362, -0.6507210, 0.6730309
6: -0.2318688, 0.3160094, -0.1639524, 0.2344593, -0.4663281, 0.4799618
7: -0.3163345, 0.3240424, -0.2429457, 0.2383102, -0.5546448, 0.5669881
8: -0.3034047, 0.3937951, -0.2266054, 0.2929426, -0.5963473, 0.6204004
9: -0.2960743, 0.3811685, -0.2162283, 0.2807564, -0.5768307, 0.5973968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6200992, upper bound: 0.6210758
time: 1.10 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6200992, upper bound: 0.6210758
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.4282873, 1.0624255, 0.5136150, 1.0505899, -0.6223027, 0.5488105
1: -0.2687270, 0.2626047, -0.2189514, 0.2150324, -0.4837594, 0.4815561
2: -0.1858643, 0.3562993, -0.1418542, 0.2998670, -0.4857312, 0.4981535
3: -0.1956849, 0.2660857, -0.1594748, 0.2213767, -0.4170615, 0.4255605
4: -0.2829364, 0.2425356, -0.2368913, 0.1905226, -0.4734590, 0.4794269
5: -0.3033190, 0.3973331, -0.2518417, 0.3306052, -0.6339242, 0.6491748
6: -0.2121295, 0.2925645, -0.1686421, 0.2409129, -0.4530424, 0.4612066
7: -0.2950160, 0.2993884, -0.2480484, 0.2450728, -0.5400888, 0.5474368
8: -0.2813072, 0.3648011, -0.2326237, 0.3009238, -0.5822309, 0.5974249
9: -0.2731100, 0.3518502, -0.2225170, 0.2872585, -0.5603685, 0.5743671

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=42, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6213716
time: 7.88 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6213716
time: 1.01 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 10.82 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6200984
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6213442
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6200984
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6213442
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6206543, upper bound: 0.6220167
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6206543, upper bound: 0.6212123
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6206543, upper bound: 0.6220167
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6206543, upper bound: 0.6212123
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6181348, upper bound: 0.6200550
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6181348, upper bound: 0.6213309
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6181348, upper bound: 0.6200550
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6181348, upper bound: 0.6213309
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6201944, upper bound: 0.6220167
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6201944, upper bound: 0.6212118
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6201944, upper bound: 0.6220167
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6201944, upper bound: 0.6212118
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6203421
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6216087
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6203421
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6216087
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6206278, upper bound: 0.6187256
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6218435, upper bound: 0.6202619
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6199196, upper bound: 0.6187256
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6206941, upper bound: 0.6202619
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6181361, upper bound: 0.6203180
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6181361, upper bound: 0.6215980
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6181361, upper bound: 0.6203180
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6181361, upper bound: 0.6215980
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6205137, upper bound: 0.6187256
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6213409, upper bound: 0.6202619
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6197221, upper bound: 0.6187256
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6202619, upper bound: 0.6202619
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6249942
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6249942
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6249942
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6249942
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6227149
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6227149
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6227149
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6227149
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6213626, upper bound: 0.6249899
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6213626, upper bound: 0.6249899
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6213626, upper bound: 0.6249899
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6213626, upper bound: 0.6249899
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6213626, upper bound: 0.6227146
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6213626, upper bound: 0.6227146
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6213626, upper bound: 0.6227146
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6213626, upper bound: 0.6227146
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6253811
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6253811
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6253811
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6253811
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6217965, upper bound: 0.6202619
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6217965, upper bound: 0.6202619
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6206941, upper bound: 0.6202619
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6206941, upper bound: 0.6202619
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6253733
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6253734
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6253734
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6253734
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6200992, upper bound: 0.6210758
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6200992, upper bound: 0.6210758
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6213716
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.82
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6213716

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.5332847, 1.0476160, 0.5616872, 1.0433842, -0.5100995, 0.4859288
1: -0.2075208, 0.2030602, -0.1910217, 0.1860027, -0.3935235, 0.3940819
2: -0.1328426, 0.2861803, -0.1200027, 0.2673327, -0.4001753, 0.4061830
3: -0.1504719, 0.2109778, -0.1373721, 0.1972418, -0.3477137, 0.3483499
4: -0.2255193, 0.1798439, -0.2095905, 0.1646237, -0.3901430, 0.3894345
5: -0.2385083, 0.3176840, -0.2190318, 0.2997580, -0.5382662, 0.5367158
6: -0.1598674, 0.2275344, -0.1483393, 0.2079921, -0.3678595, 0.3758737
7: -0.2376458, 0.2315206, -0.2226892, 0.2123595, -0.4500054, 0.4542097
8: -0.2206139, 0.2843786, -0.2037058, 0.2602109, -0.4808248, 0.4880844
9: -0.2101207, 0.2741386, -0.1928850, 0.2554634, -0.4655841, 0.4670236

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=46, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6198216, upper bound: 0.6215045
time: 1.21 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6198216, upper bound: 0.6215045
time: 1.25 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.5365080, 1.0471358, 0.5473926, 1.0455141, -0.5090061, 0.4997432
1: -0.2056483, 0.2011244, -0.1993254, 0.1945876, -0.4002359, 0.4004498
2: -0.1313855, 0.2840413, -0.1264649, 0.2768183, -0.4082038, 0.4105062
3: -0.1489853, 0.2094189, -0.1439651, 0.2041550, -0.3531402, 0.3533840
4: -0.2237116, 0.1781167, -0.2176072, 0.1722839, -0.3959954, 0.3957239
5: -0.2362981, 0.3156496, -0.2288340, 0.3087799, -0.5450780, 0.5444836
6: -0.1585591, 0.2253165, -0.1541412, 0.2178274, -0.3763865, 0.3794578
7: -0.2359485, 0.2293461, -0.2302166, 0.2220029, -0.4579514, 0.4595628
8: -0.2186950, 0.2816359, -0.2122153, 0.2723742, -0.4910692, 0.4938512
9: -0.2081648, 0.2720194, -0.2015595, 0.2648624, -0.4730272, 0.4735789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6198216, upper bound: 0.6225501
time: 4.55 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6198216, upper bound: 0.6225501
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.5040902, 1.0519111, 0.5616872, 1.0433842, -0.5392939, 0.4902239
1: -0.2245077, 0.2203427, -0.1910217, 0.1860027, -0.4105104, 0.4113644
2: -0.1467668, 0.3061663, -0.1200027, 0.2673327, -0.4140996, 0.4261690
3: -0.1635167, 0.2263674, -0.1373721, 0.1972418, -0.3607586, 0.3637395
4: -0.2420311, 0.1963286, -0.2095905, 0.1646237, -0.4066549, 0.4059191
5: -0.2575880, 0.3380537, -0.2190318, 0.2997580, -0.5573460, 0.5570856
6: -0.1734964, 0.2466787, -0.1483393, 0.2079921, -0.3814886, 0.3950180
7: -0.2532912, 0.2511359, -0.2226892, 0.2123595, -0.4656508, 0.4738250
8: -0.2380579, 0.3080543, -0.2037058, 0.2602109, -0.4982689, 0.5117600
9: -0.2281645, 0.2944687, -0.1928850, 0.2554634, -0.4836279, 0.4873537

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=46, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6200984
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6200984
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.5076016, 1.0514241, 0.5473926, 1.0455141, -0.5379125, 0.5040315
1: -0.2224592, 0.2183849, -0.1993254, 0.1945876, -0.4170468, 0.4177102
2: -0.1449557, 0.3038440, -0.1264649, 0.2768183, -0.4217740, 0.4303088
3: -0.1620266, 0.2245274, -0.1439651, 0.2041550, -0.3661816, 0.3684925
4: -0.2401361, 0.1941880, -0.2176072, 0.1722839, -0.4124199, 0.4117953
5: -0.2554694, 0.3353077, -0.2288340, 0.3087799, -0.5642493, 0.5641418
6: -0.1717067, 0.2445528, -0.1541412, 0.2178274, -0.3895341, 0.3986941
7: -0.2513583, 0.2489007, -0.2302166, 0.2220029, -0.4733613, 0.4791173
8: -0.2360545, 0.3054253, -0.2122153, 0.2723742, -0.5084287, 0.5176407
9: -0.2260824, 0.2918105, -0.2015595, 0.2648624, -0.4909448, 0.4933700

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6213442
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6213442
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.5548712, 1.0444001, 0.5583270, 1.0438850, -0.4890138, 0.4860731
1: -0.1949811, 0.1900961, -0.1929736, 0.1880208, -0.3830019, 0.3830698
2: -0.1230840, 0.2718558, -0.1215217, 0.2695625, -0.3926466, 0.3933775
3: -0.1405158, 0.2005381, -0.1389220, 0.1988669, -0.3393827, 0.3394601
4: -0.2134131, 0.1682763, -0.2114750, 0.1664245, -0.3798376, 0.3797513
5: -0.2237059, 0.3040599, -0.2213362, 0.3018788, -0.5255846, 0.5253961
6: -0.1511058, 0.2126819, -0.1497032, 0.2103041, -0.3614099, 0.3623851
7: -0.2262785, 0.2169579, -0.2244588, 0.2146266, -0.4409050, 0.4414166
8: -0.2077634, 0.2660105, -0.2057062, 0.2630700, -0.4708335, 0.4717167
9: -0.1970212, 0.2599451, -0.1949242, 0.2576728, -0.4546940, 0.4548692

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6207517, upper bound: 0.6222330
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6220421, upper bound: 0.6229761
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.5592567, 1.0437465, 0.5058676, 1.0517009, -0.4924443, 0.5378789
1: -0.1924335, 0.1874624, -0.2234473, 0.2195259, -0.4119594, 0.4109097
2: -0.1211014, 0.2689456, -0.1452370, 0.3043738, -0.4254752, 0.4141826
3: -0.1384931, 0.1984172, -0.1631170, 0.2242371, -0.3627302, 0.3615341
4: -0.2109535, 0.1659262, -0.2408955, 0.1945360, -0.4054894, 0.4068217
5: -0.2206986, 0.3012920, -0.2573088, 0.3349879, -0.5556865, 0.5586008
6: -0.1493258, 0.2096643, -0.1709954, 0.2463986, -0.3957245, 0.3806598
7: -0.2239690, 0.2139993, -0.2520834, 0.2500166, -0.4739857, 0.4660826
8: -0.2051525, 0.2622787, -0.2369352, 0.3077078, -0.5128603, 0.4992140
9: -0.1943599, 0.2570614, -0.2267585, 0.2921661, -0.4865260, 0.4838199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6207517, upper bound: 0.6212997
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6220421, upper bound: 0.6216394
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.5289317, 1.0482646, 0.5583270, 1.0438850, -0.5149533, 0.4899377
1: -0.2100493, 0.2056745, -0.1929736, 0.1880208, -0.3980701, 0.3986482
2: -0.1348104, 0.2890689, -0.1215217, 0.2695625, -0.4043730, 0.4105906
3: -0.1524796, 0.2130829, -0.1389220, 0.1988669, -0.3513465, 0.3520049
4: -0.2279605, 0.1821765, -0.2114750, 0.1664245, -0.3943850, 0.3936515
5: -0.2414932, 0.3204312, -0.2213362, 0.3018788, -0.5433720, 0.5417675
6: -0.1616341, 0.2305295, -0.1497032, 0.2103041, -0.3719382, 0.3802328
7: -0.2399380, 0.2344570, -0.2244588, 0.2146266, -0.4545646, 0.4589158
8: -0.2232052, 0.2880825, -0.2057062, 0.2630700, -0.4862752, 0.4937887
9: -0.2127623, 0.2770008, -0.1949242, 0.2576728, -0.4704351, 0.4719250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6187270, upper bound: 0.6207624
time: 1.17 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6206941, upper bound: 0.6220167
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.5330682, 1.0476483, 0.5058676, 1.0517009, -0.5186328, 0.5417807
1: -0.2076466, 0.2031904, -0.2234473, 0.2195259, -0.4271725, 0.4266376
2: -0.1329405, 0.2863241, -0.1452370, 0.3043738, -0.4373144, 0.4315611
3: -0.1505717, 0.2110826, -0.1631170, 0.2242371, -0.3748088, 0.3741996
4: -0.2256408, 0.1799600, -0.2408955, 0.1945360, -0.4201768, 0.4208555
5: -0.2386569, 0.3178206, -0.2573088, 0.3349879, -0.5736448, 0.5751295
6: -0.1599553, 0.2276834, -0.1709954, 0.2463986, -0.4063540, 0.3986788
7: -0.2377599, 0.2316666, -0.2520834, 0.2500166, -0.4877765, 0.4837500
8: -0.2207429, 0.2845628, -0.2369352, 0.3077078, -0.5284507, 0.5214980
9: -0.2102521, 0.2742811, -0.2267585, 0.2921661, -0.5024182, 0.5010396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6187270, upper bound: 0.6201649
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6206941, upper bound: 0.6212123
time: 1.55 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.4775736, 1.0555892, 0.5618162, 1.0433652, -0.5657917, 0.4937730
1: -0.2399760, 0.2351264, -0.1909468, 0.1859253, -0.4259013, 0.4260733
2: -0.1604435, 0.3237034, -0.1199444, 0.2672471, -0.4276906, 0.4436478
3: -0.1747695, 0.2402613, -0.1373127, 0.1971794, -0.3719488, 0.3775740
4: -0.2563401, 0.2124922, -0.2095181, 0.1645547, -0.4208948, 0.4220103
5: -0.2735851, 0.3587903, -0.2189435, 0.2996767, -0.5732617, 0.5777338
6: -0.1870107, 0.2627299, -0.1482870, 0.2079035, -0.3949142, 0.4110169
7: -0.2678870, 0.2680151, -0.2226213, 0.2122727, -0.4801597, 0.4906364
8: -0.2531870, 0.3279046, -0.2036290, 0.2601011, -0.5132881, 0.5315336
9: -0.2438869, 0.3145413, -0.1928068, 0.2553785, -0.4992654, 0.5073481

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6196607, upper bound: 0.6214766
time: 1.34 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6196607, upper bound: 0.6214766
time: 1.28 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.4889469, 1.0540119, 0.5471380, 1.0455520, -0.5566052, 0.5068740
1: -0.2333414, 0.2287856, -0.1994734, 0.1947404, -0.4280819, 0.4282589
2: -0.1545775, 0.3161814, -0.1265799, 0.2769874, -0.4315649, 0.4427614
3: -0.1699430, 0.2343020, -0.1440825, 0.2042779, -0.3742210, 0.3783845
4: -0.2502029, 0.2055595, -0.2177500, 0.1724203, -0.4226232, 0.4233095
5: -0.2667238, 0.3498962, -0.2290087, 0.3089406, -0.5756644, 0.5789049
6: -0.1812143, 0.2558452, -0.1542446, 0.2180027, -0.3992169, 0.4100898
7: -0.2616266, 0.2607754, -0.2303507, 0.2221749, -0.4838016, 0.4911261
8: -0.2466980, 0.3193906, -0.2123669, 0.2725908, -0.5192888, 0.5317576
9: -0.2371433, 0.3059319, -0.2017140, 0.2650298, -0.5021731, 0.5076458

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6196607, upper bound: 0.6225322
time: 1.30 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6196607, upper bound: 0.6225322
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.4543309, 1.0588131, 0.5618162, 1.0433652, -0.5890344, 0.4969969
1: -0.2535346, 0.2480848, -0.1909468, 0.1859253, -0.4394599, 0.4390316
2: -0.1724315, 0.3390751, -0.1199444, 0.2672471, -0.4396787, 0.4590195
3: -0.1846329, 0.2524397, -0.1373127, 0.1971794, -0.3818123, 0.3897524
4: -0.2688825, 0.2266602, -0.2095181, 0.1645547, -0.4334372, 0.4361784
5: -0.2876073, 0.3769665, -0.2189435, 0.2996767, -0.5872840, 0.5959100
6: -0.1988564, 0.2767995, -0.1482870, 0.2079035, -0.4067599, 0.4250865
7: -0.2806805, 0.2828103, -0.2226213, 0.2122727, -0.4929532, 0.5054316
8: -0.2664481, 0.3453044, -0.2036290, 0.2601011, -0.5265492, 0.5489334
9: -0.2576680, 0.3321357, -0.1928068, 0.2553785, -0.5130465, 0.5249425

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6181361, upper bound: 0.6200550
time: 1.74 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6181361, upper bound: 0.6200550
time: 1.26 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.4647490, 1.0573682, 0.5471380, 1.0455520, -0.5808030, 0.5102302
1: -0.2474573, 0.2422764, -0.1994734, 0.1947404, -0.4421977, 0.4417498
2: -0.1670582, 0.3321852, -0.1265799, 0.2769874, -0.4440456, 0.4587651
3: -0.1802118, 0.2469811, -0.1440825, 0.2042779, -0.3844898, 0.3910635
4: -0.2632607, 0.2203098, -0.2177500, 0.1724203, -0.4356810, 0.4380598
5: -0.2813222, 0.3688194, -0.2290087, 0.3089406, -0.5902628, 0.5978281
6: -0.1935468, 0.2704933, -0.1542446, 0.2180027, -0.4115495, 0.4247378
7: -0.2749462, 0.2761785, -0.2303507, 0.2221749, -0.4971211, 0.5065293
8: -0.2605041, 0.3375054, -0.2123669, 0.2725908, -0.5330948, 0.5498724
9: -0.2514910, 0.3242494, -0.2017140, 0.2650298, -0.5165207, 0.5259634

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6181361, upper bound: 0.6213309
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6181361, upper bound: 0.6213309
time: 2.49 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.5044389, 1.0518630, 0.5582559, 1.0438956, -0.5394567, 0.4936070
1: -0.2243043, 0.2201482, -0.1930149, 0.1880634, -0.4123677, 0.4131631
2: -0.1465871, 0.3059356, -0.1215538, 0.2696097, -0.4161968, 0.4274894
3: -0.1633688, 0.2261846, -0.1389547, 0.1989012, -0.3622700, 0.3651393
4: -0.2418429, 0.1961161, -0.2115148, 0.1664625, -0.4083054, 0.4076309
5: -0.2573777, 0.3377811, -0.2213848, 0.3019236, -0.5593013, 0.5591658
6: -0.1733187, 0.2464676, -0.1497320, 0.2103529, -0.3836716, 0.3961996
7: -0.2530992, 0.2509140, -0.2244961, 0.2146744, -0.4677736, 0.4754101
8: -0.2378591, 0.3077933, -0.2057484, 0.2631304, -0.5009894, 0.5135416
9: -0.2279577, 0.2942047, -0.1949671, 0.2577195, -0.4856772, 0.4891719

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6205192, upper bound: 0.6222430
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6216394, upper bound: 0.6229761
time: 1.15 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.5092793, 1.0511917, 0.5058234, 1.0517074, -0.5424281, 0.5453683
1: -0.2214807, 0.2174498, -0.2234731, 0.2195525, -0.4410332, 0.4409229
2: -0.1440906, 0.3027345, -0.1452571, 0.3044032, -0.4484937, 0.4479916
3: -0.1613148, 0.2236486, -0.1631375, 0.2242585, -0.3855733, 0.3867861
4: -0.2392310, 0.1931656, -0.2409203, 0.1945598, -0.4337908, 0.4340859
5: -0.2544576, 0.3339959, -0.2573393, 0.3350160, -0.5894735, 0.5913352
6: -0.1708519, 0.2435375, -0.1710134, 0.2464291, -0.4172810, 0.4145509
7: -0.2504352, 0.2478330, -0.2521068, 0.2500465, -0.5004817, 0.4999398
8: -0.2350975, 0.3041698, -0.2369616, 0.3077456, -0.5428431, 0.5411313
9: -0.2250879, 0.2905407, -0.2267853, 0.2921951, -0.5172830, 0.5173260

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6205192, upper bound: 0.6213032
time: 1.25 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6216394, upper bound: 0.6216394
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.4813803, 1.0550613, 0.5582559, 1.0438956, -0.5625153, 0.4968054
1: -0.2377555, 0.2330040, -0.1930149, 0.1880634, -0.4258189, 0.4260189
2: -0.1584801, 0.3211858, -0.1215538, 0.2696097, -0.4280898, 0.4427396
3: -0.1731541, 0.2382666, -0.1389547, 0.1989012, -0.3720553, 0.3772213
4: -0.2542860, 0.2101718, -0.2115148, 0.1664625, -0.4207484, 0.4216866
5: -0.2712886, 0.3558133, -0.2213848, 0.3019236, -0.5732122, 0.5771981
6: -0.1850706, 0.2604257, -0.1497320, 0.2103529, -0.3954235, 0.4101577
7: -0.2657915, 0.2655919, -0.2244961, 0.2146744, -0.4804659, 0.4900880
8: -0.2510152, 0.3250551, -0.2057484, 0.2631304, -0.5141456, 0.5308034
9: -0.2416298, 0.3116598, -0.1949671, 0.2577195, -0.4993493, 0.5066268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6187256, upper bound: 0.6207901
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6202619, upper bound: 0.6220167
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.4860020, 1.0544202, 0.5058234, 1.0517074, -0.5657054, 0.5485969
1: -0.2350592, 0.2304273, -0.2234731, 0.2195525, -0.4546117, 0.4539004
2: -0.1560962, 0.3181291, -0.1452571, 0.3044032, -0.4604994, 0.4633862
3: -0.1711927, 0.2358449, -0.1631375, 0.2242585, -0.3954512, 0.3989825
4: -0.2517920, 0.2073545, -0.2409203, 0.1945598, -0.4463517, 0.4482748
5: -0.2685004, 0.3521990, -0.2573393, 0.3350160, -0.6035163, 0.6095383
6: -0.1827150, 0.2576280, -0.1710134, 0.2464291, -0.4291441, 0.4286414
7: -0.2632476, 0.2626498, -0.2521068, 0.2500465, -0.5132941, 0.5147566
8: -0.2483781, 0.3215951, -0.2369616, 0.3077456, -0.5561237, 0.5585567
9: -0.2388894, 0.3081610, -0.2267853, 0.2921951, -0.5310845, 0.5349463

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6187256, upper bound: 0.6201923
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6202619, upper bound: 0.6212118
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.5332397, 1.0476227, 0.5336711, 1.0475584, -0.5143187, 0.5139515
1: -0.2075468, 0.2030872, -0.2072963, 0.2028283, -0.4103751, 0.4103835
2: -0.1328629, 0.2862101, -0.1326680, 0.2859240, -0.4187868, 0.4188782
3: -0.1504925, 0.2109995, -0.1502937, 0.2107910, -0.3612836, 0.3612931
4: -0.2255445, 0.1798680, -0.2253026, 0.1796370, -0.4051815, 0.4051706
5: -0.2385391, 0.3177123, -0.2382435, 0.3174401, -0.5559793, 0.5559558
6: -0.1598856, 0.2275652, -0.1597106, 0.2272685, -0.3871541, 0.3872758
7: -0.2376694, 0.2315509, -0.2374423, 0.2312599, -0.4689294, 0.4689932
8: -0.2206406, 0.2844167, -0.2203838, 0.2840499, -0.5046905, 0.5048006
9: -0.2101480, 0.2741682, -0.2098863, 0.2738847, -0.4840327, 0.4840546

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6252939, upper bound: 0.6223945
time: 1.18 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6236611, upper bound: 0.6223945
time: 1.19 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.5369794, 1.0470656, 0.5160234, 1.0502460, -0.5132666, 0.5310422
1: -0.2053745, 0.2008412, -0.2175479, 0.2136490, -0.4190235, 0.4183891
2: -0.1311723, 0.2837283, -0.1406460, 0.2982261, -0.4293984, 0.4243743
3: -0.1487678, 0.2091909, -0.1584331, 0.2200767, -0.3688444, 0.3676240
4: -0.2234471, 0.1778641, -0.2355523, 0.1890938, -0.4125409, 0.4134164
5: -0.2359747, 0.3153520, -0.2503449, 0.3287858, -0.5647604, 0.5656969
6: -0.1583678, 0.2249922, -0.1674260, 0.2394109, -0.3977787, 0.3924183
7: -0.2357001, 0.2290280, -0.2467355, 0.2434935, -0.4791936, 0.4757636
8: -0.2184143, 0.2812347, -0.2312109, 0.2990664, -0.5174807, 0.5124457
9: -0.2078787, 0.2717092, -0.2210458, 0.2854886, -0.4933673, 0.4927550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6255694, upper bound: 0.6236558
time: 1.19 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6236611, upper bound: 0.6236558
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.5793729, 1.0407494, 0.5336711, 1.0475584, -0.4681855, 0.5070783
1: -0.1807481, 0.1753813, -0.2072963, 0.2028283, -0.3835764, 0.3826776
2: -0.1120075, 0.2555968, -0.1326680, 0.2859240, -0.3979314, 0.3882648
3: -0.1292152, 0.1886887, -0.1502937, 0.2107910, -0.3400062, 0.3389823
4: -0.1996719, 0.1551465, -0.2253026, 0.1796370, -0.3793089, 0.3804491
5: -0.2069043, 0.2885959, -0.2382435, 0.3174401, -0.5243444, 0.5268393
6: -0.1411611, 0.1958234, -0.1597106, 0.2272685, -0.3684295, 0.3555341
7: -0.2133760, 0.2004286, -0.2374423, 0.2312599, -0.4446359, 0.4378709
8: -0.1931774, 0.2451620, -0.2203838, 0.2840499, -0.4772273, 0.4655458
9: -0.1821526, 0.2438345, -0.2098863, 0.2738847, -0.4560373, 0.4537209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6182497, upper bound: 0.6185786
time: 1.30 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6171597, upper bound: 0.6185352
time: 1.26 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.5825190, 1.0402807, 0.5160234, 1.0502460, -0.4677270, 0.5242573
1: -0.1789205, 0.1734919, -0.2175479, 0.2136490, -0.3925695, 0.3910398
2: -0.1105852, 0.2535091, -0.1406460, 0.2982261, -0.4088114, 0.3941551
3: -0.1277641, 0.1871673, -0.1584331, 0.2200767, -0.3478408, 0.3456004
4: -0.1979074, 0.1534605, -0.2355523, 0.1890938, -0.3870012, 0.3890129
5: -0.2047470, 0.2866102, -0.2503449, 0.3287858, -0.5335327, 0.5369551
6: -0.1398842, 0.1936586, -0.1674260, 0.2394109, -0.3792951, 0.3610846
7: -0.2117193, 0.1983062, -0.2467355, 0.2434935, -0.4552128, 0.4450417
8: -0.1913047, 0.2424849, -0.2312109, 0.2990664, -0.4903711, 0.4736958
9: -0.1802434, 0.2417660, -0.2210458, 0.2854886, -0.4657320, 0.4628118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6184171, upper bound: 0.6201433
time: 1.19 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6171597, upper bound: 0.6201143
time: 1.17 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.5362493, 1.0471743, 0.4926630, 1.0536684, -0.5174190, 0.5545113
1: -0.2057985, 0.2012796, -0.2311179, 0.2274561, -0.4332546, 0.4323974
2: -0.1315023, 0.2842128, -0.1512064, 0.3131364, -0.4446387, 0.4354193
3: -0.1491044, 0.2095440, -0.1692072, 0.2306231, -0.3797275, 0.3787512
4: -0.2238565, 0.1782552, -0.2483010, 0.2016120, -0.4254685, 0.4265562
5: -0.2364753, 0.3158127, -0.2663636, 0.3433218, -0.5797970, 0.5821763
6: -0.1586641, 0.2254944, -0.1763549, 0.2554839, -0.4141480, 0.4018493
7: -0.2360845, 0.2295205, -0.2590369, 0.2589247, -0.4950091, 0.4885574
8: -0.2188489, 0.2818559, -0.2447960, 0.3189436, -0.5377924, 0.5266520
9: -0.2083216, 0.2721892, -0.2347715, 0.3008485, -0.5091701, 0.5069607

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6155709, upper bound: 0.6152664
time: 1.23 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6171691, upper bound: 0.6152664
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.5270282, 1.0485482, 0.5314887, 1.0478835, -0.5208553, 0.5170596
1: -0.2111552, 0.2068177, -0.2085640, 0.2041389, -0.4152941, 0.4153817
2: -0.1356710, 0.2903321, -0.1336545, 0.2873721, -0.4230431, 0.4239866
3: -0.1533575, 0.2140035, -0.1513002, 0.2118463, -0.3652038, 0.3653037
4: -0.2290281, 0.1831966, -0.2265265, 0.1808064, -0.4098344, 0.4097231
5: -0.2427985, 0.3216326, -0.2397399, 0.3188174, -0.5616159, 0.5613725
6: -0.1624067, 0.2318393, -0.1605963, 0.2287701, -0.3911768, 0.3924356
7: -0.2409404, 0.2357413, -0.2385916, 0.2327320, -0.4736724, 0.4743328
8: -0.2243384, 0.2897023, -0.2216831, 0.2859067, -0.5102452, 0.5113854
9: -0.2139174, 0.2782525, -0.2112107, 0.2753196, -0.4892371, 0.4894632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6173258, upper bound: 0.6170159
time: 9.90 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6183778, upper bound: 0.6170159
time: 1.32 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.4784256, 1.0554712, 0.4965693, 1.0530863, -0.5746607, 0.5589019
1: -0.2394790, 0.2346514, -0.2288487, 0.2251101, -0.4645891, 0.4635001
2: -0.1600041, 0.3231399, -0.1494406, 0.3105443, -0.4705483, 0.4725804
3: -0.1744079, 0.2398147, -0.1674057, 0.2287339, -0.4031419, 0.4072204
4: -0.2558804, 0.2119729, -0.2461102, 0.1995188, -0.4553992, 0.4580831
5: -0.2730711, 0.3581240, -0.2636850, 0.3408565, -0.6139275, 0.6218090
6: -0.1865764, 0.2622141, -0.1747694, 0.2527964, -0.4393727, 0.4369836
7: -0.2674179, 0.2674727, -0.2569798, 0.2562894, -0.5237073, 0.5244525
8: -0.2527008, 0.3272670, -0.2424707, 0.3156198, -0.5683206, 0.5697377
9: -0.2433817, 0.3138963, -0.2324011, 0.2982800, -0.5416617, 0.5462974

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6148784, upper bound: 0.6152647
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6164195, upper bound: 0.6152664
time: 1.43 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.4681116, 1.0569017, 0.5356633, 1.0472620, -0.5791503, 0.5212384
1: -0.2454956, 0.2404017, -0.2061389, 0.2016317, -0.4471273, 0.4465406
2: -0.1653237, 0.3299610, -0.1317673, 0.2846017, -0.4499255, 0.4617283
3: -0.1787848, 0.2452191, -0.1493748, 0.2098274, -0.3886123, 0.3945938
4: -0.2614461, 0.2182599, -0.2241853, 0.1785692, -0.4400153, 0.4424452
5: -0.2792934, 0.3661896, -0.2368772, 0.3161827, -0.5954762, 0.6030667
6: -0.1918330, 0.2684575, -0.1589019, 0.2258977, -0.4177306, 0.4273594
7: -0.2730951, 0.2740380, -0.2363932, 0.2299158, -0.5030109, 0.5104312
8: -0.2585856, 0.3349882, -0.2191979, 0.2823545, -0.5409401, 0.5541861
9: -0.2494972, 0.3217038, -0.2086774, 0.2725747, -0.5220719, 0.5303812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6162086, upper bound: 0.6170155
time: 2.39 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6171597, upper bound: 0.6170159
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.4873266, 1.0542364, 0.5335602, 1.0475749, -0.5602483, 0.5206763
1: -0.2342866, 0.2296887, -0.2073607, 0.2028948, -0.4371814, 0.4370494
2: -0.1554130, 0.3172531, -0.1327180, 0.2859975, -0.4414104, 0.4499711
3: -0.1706306, 0.2351509, -0.1503448, 0.2108445, -0.3814750, 0.3854957
4: -0.2510770, 0.2065471, -0.2253647, 0.1796963, -0.4307733, 0.4319118
5: -0.2677011, 0.3511631, -0.2383193, 0.3175101, -0.5852112, 0.5894824
6: -0.1820400, 0.2568261, -0.1597555, 0.2273448, -0.4093848, 0.4165816
7: -0.2625184, 0.2618066, -0.2375007, 0.2313347, -0.4938531, 0.4993073
8: -0.2476223, 0.3206035, -0.2204498, 0.2841442, -0.5317665, 0.5410534
9: -0.2381040, 0.3071582, -0.2099536, 0.2739576, -0.5120616, 0.5171118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6252907, upper bound: 0.6223945
time: 1.21 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6236558, upper bound: 0.6223945
time: 1.16 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.4993103, 1.0525740, 0.5156627, 1.0503029, -0.5509926, 0.5369113
1: -0.2272960, 0.2230076, -0.2177573, 0.2138778, -0.4411738, 0.4407648
2: -0.1492322, 0.3093275, -0.1408091, 0.2984970, -0.4477293, 0.4501366
3: -0.1655451, 0.2288719, -0.1585995, 0.2202914, -0.3858365, 0.3874713
4: -0.2446105, 0.1992423, -0.2357735, 0.1892871, -0.4338975, 0.4350158
5: -0.2604716, 0.3417917, -0.2505921, 0.3290245, -0.5894961, 0.5923837
6: -0.1759325, 0.2495720, -0.1676021, 0.2396591, -0.4155916, 0.4171741
7: -0.2559223, 0.2541786, -0.2469254, 0.2437544, -0.4996767, 0.5011040
8: -0.2407852, 0.3116324, -0.2314428, 0.2993733, -0.5401584, 0.5430752
9: -0.2309986, 0.2980869, -0.2212890, 0.2857256, -0.5167242, 0.5193759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6255477, upper bound: 0.6236558
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6236558, upper bound: 0.6236558
time: 1.20 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.5307614, 1.0479921, 0.5335602, 1.0475749, -0.5168135, 0.5144320
1: -0.2089866, 0.2045756, -0.2073607, 0.2028948, -0.4118813, 0.4119363
2: -0.1339833, 0.2878546, -0.1327180, 0.2859975, -0.4199808, 0.4205727
3: -0.1516356, 0.2121981, -0.1503448, 0.2108445, -0.3624801, 0.3625429
4: -0.2269344, 0.1811960, -0.2253647, 0.1796963, -0.4066307, 0.4065607
5: -0.2402385, 0.3192765, -0.2383193, 0.3175101, -0.5577486, 0.5575958
6: -0.1608915, 0.2292706, -0.1597555, 0.2273448, -0.3882363, 0.3890261
7: -0.2389745, 0.2332228, -0.2375007, 0.2313347, -0.4703092, 0.4707235
8: -0.2221159, 0.2865255, -0.2204498, 0.2841442, -0.5062601, 0.5069753
9: -0.2116519, 0.2757978, -0.2099536, 0.2739576, -0.4856095, 0.4857515

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6180263, upper bound: 0.6185691
time: 1.31 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6170159, upper bound: 0.6185263
time: 1.18 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.5387831, 1.0467968, 0.5156627, 1.0503029, -0.5115197, 0.5311341
1: -0.2043267, 0.1997582, -0.2177573, 0.2138778, -0.4182044, 0.4175155
2: -0.1303569, 0.2825315, -0.1408091, 0.2984970, -0.4288539, 0.4233405
3: -0.1479359, 0.2083187, -0.1585995, 0.2202914, -0.3682274, 0.3669181
4: -0.2224357, 0.1768975, -0.2357735, 0.1892871, -0.4117227, 0.4126710
5: -0.2347378, 0.3142137, -0.2505921, 0.3290245, -0.5637623, 0.5648057
6: -0.1576357, 0.2237511, -0.1676021, 0.2396591, -0.3972948, 0.3913532
7: -0.2347503, 0.2278112, -0.2469254, 0.2437544, -0.4785047, 0.4747366
8: -0.2173406, 0.2796999, -0.2314428, 0.2993733, -0.5167139, 0.5111427
9: -0.2067841, 0.2705234, -0.2212890, 0.2857256, -0.4925096, 0.4918124

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6181335, upper bound: 0.6201294
time: 1.18 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6170159, upper bound: 0.6201044
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.4885482, 1.0540669, 0.4927876, 1.0536497, -0.5651015, 0.5612793
1: -0.2335740, 0.2290076, -0.2310456, 0.2273814, -0.4609554, 0.4600531
2: -0.1547830, 0.3164451, -0.1511502, 0.3130537, -0.4678367, 0.4675952
3: -0.1701121, 0.2345109, -0.1691498, 0.2305629, -0.4006750, 0.4036607
4: -0.2504179, 0.2058025, -0.2482311, 0.2015453, -0.4519632, 0.4540335
5: -0.2669642, 0.3502077, -0.2662783, 0.3432432, -0.6102074, 0.6164860
6: -0.1814173, 0.2560866, -0.1763044, 0.2553983, -0.4368156, 0.4323909
7: -0.2618460, 0.2610290, -0.2589714, 0.2588405, -0.5206866, 0.5200005
8: -0.2469253, 0.3196890, -0.2447218, 0.3188377, -0.5657631, 0.5644108
9: -0.2373797, 0.3062336, -0.2346962, 0.3007665, -0.5381462, 0.5409298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6150744, upper bound: 0.6152567
time: 1.22 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6171072, upper bound: 0.6152664
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.4785240, 1.0554572, 0.5312068, 1.0479257, -0.5694017, 0.5242504
1: -0.2394216, 0.2345965, -0.2087277, 0.2043081, -0.4437298, 0.4433242
2: -0.1599533, 0.3230748, -0.1337819, 0.2875592, -0.4475124, 0.4568567
3: -0.1743661, 0.2397633, -0.1514302, 0.2119827, -0.3863488, 0.3911935
4: -0.2558272, 0.2119129, -0.2266846, 0.1809574, -0.4367846, 0.4385974
5: -0.2730117, 0.3580470, -0.2399332, 0.3189952, -0.5920069, 0.5979801
6: -0.1865263, 0.2621545, -0.1607107, 0.2289641, -0.4154903, 0.4228652
7: -0.2673638, 0.2674101, -0.2387400, 0.2329222, -0.5002860, 0.5061501
8: -0.2526447, 0.3271932, -0.2218508, 0.2861467, -0.5387914, 0.5490440
9: -0.2433234, 0.3138218, -0.2113818, 0.2755049, -0.5188283, 0.5252035

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6164850, upper bound: 0.6170137
time: 1.23 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6181180, upper bound: 0.6170159
time: 1.22 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.4330829, 1.0617602, 0.4966249, 1.0530779, -0.6199950, 0.5651353
1: -0.2659294, 0.2599310, -0.2288165, 0.2250767, -0.4910061, 0.4887475
2: -0.1833907, 0.3531275, -0.1494155, 0.3105074, -0.4938982, 0.5025430
3: -0.1936496, 0.2635729, -0.1673801, 0.2287071, -0.4223567, 0.4309529
4: -0.2803484, 0.2396123, -0.2460791, 0.1994891, -0.4798374, 0.4856914
5: -0.3004258, 0.3935827, -0.2636470, 0.3408214, -0.6412473, 0.6572297
6: -0.2096854, 0.2896615, -0.1747469, 0.2527581, -0.4624436, 0.4644083
7: -0.2923762, 0.2963356, -0.2569506, 0.2562521, -0.5486283, 0.5532862
8: -0.2785709, 0.3612107, -0.2424375, 0.3155727, -0.5941436, 0.6036482
9: -0.2702665, 0.3482198, -0.2323675, 0.2982435, -0.5685100, 0.5805873

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6143194, upper bound: 0.6152486
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6163938, upper bound: 0.6152664
time: 1.23 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.4229195, 1.0631701, 0.5352825, 1.0473185, -0.6243989, 0.5278876
1: -0.2718582, 0.2655973, -0.2063602, 0.2018604, -0.4737187, 0.4719575
2: -0.1886328, 0.3598494, -0.1319394, 0.2848546, -0.4734874, 0.4917888
3: -0.1979627, 0.2688982, -0.1495505, 0.2100115, -0.4079742, 0.4184487
4: -0.2858328, 0.2458076, -0.2243989, 0.1787734, -0.4646062, 0.4702065
5: -0.3065573, 0.4015307, -0.2371384, 0.3164229, -0.6229802, 0.6386691
6: -0.2148653, 0.2958137, -0.1590565, 0.2261598, -0.4410251, 0.4548702
7: -0.2979705, 0.3028052, -0.2365937, 0.2301727, -0.5281432, 0.5393990
8: -0.2843696, 0.3688192, -0.2194245, 0.2826788, -0.5670484, 0.5882437
9: -0.2762926, 0.3559134, -0.2089084, 0.2728252, -0.5491179, 0.5648218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6151921, upper bound: 0.6170130
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6170159, upper bound: 0.6170159
time: 1.19 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.4588694, 1.0581836, 0.5289122, 1.0482676, -0.5893982, 0.5292714
1: -0.2508871, 0.2455544, -0.2100608, 0.2056862, -0.4565733, 0.4556152
2: -0.1700907, 0.3360735, -0.1348193, 0.2890818, -0.4591725, 0.4708928
3: -0.1827069, 0.2500617, -0.1524885, 0.2130924, -0.3957992, 0.4025502
4: -0.2664334, 0.2238937, -0.2279716, 0.1821870, -0.4486204, 0.4518653
5: -0.2848692, 0.3734171, -0.2415066, 0.3204436, -0.6053128, 0.6149238
6: -0.1965434, 0.2740521, -0.1616421, 0.2305429, -0.4270862, 0.4356942
7: -0.2781824, 0.2799212, -0.2399483, 0.2344704, -0.5126529, 0.5198696
8: -0.2638585, 0.3419070, -0.2232168, 0.2880992, -0.5519577, 0.5651238
9: -0.2549771, 0.3287001, -0.2127742, 0.2770138, -0.5319910, 0.5414743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6236159, upper bound: 0.6261992
time: 1.13 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6236159, upper bound: 0.6261992
time: 1.30 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.4588694, 1.0581836, 0.4824756, 1.0549092, -0.5960398, 0.5757080
1: -0.2508871, 0.2455544, -0.2371165, 0.2323934, -0.4832805, 0.4826710
2: -0.1700907, 0.3360735, -0.1579152, 0.3204614, -0.4905521, 0.4939887
3: -0.1827069, 0.2500617, -0.1726893, 0.2376928, -0.4203997, 0.4227509
4: -0.2664334, 0.2238937, -0.2536948, 0.2095042, -0.4759376, 0.4775885
5: -0.2848692, 0.3734171, -0.2706279, 0.3549570, -0.6398262, 0.6440450
6: -0.1965434, 0.2740521, -0.1845124, 0.2597626, -0.4563060, 0.4585645
7: -0.2781824, 0.2799212, -0.2651888, 0.2648947, -0.5430772, 0.5451100
8: -0.2638585, 0.3419070, -0.2503902, 0.3242351, -0.5880936, 0.5922971
9: -0.2549771, 0.3287001, -0.2409804, 0.3108306, -0.5658077, 0.5696805

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6236159, upper bound: 0.6261992
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6236159, upper bound: 0.6261992
time: 1.23 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.4234510, 1.0630963, 0.5289122, 1.0482676, -0.6248167, 0.5341841
1: -0.2715483, 0.2653010, -0.2100608, 0.2056862, -0.4772345, 0.4753618
2: -0.1883587, 0.3594979, -0.1348193, 0.2890818, -0.4774405, 0.4943172
3: -0.1977373, 0.2686198, -0.1524885, 0.2130924, -0.4108297, 0.4211084
4: -0.2855461, 0.2454836, -0.2279716, 0.1821870, -0.4677331, 0.4734552
5: -0.3062367, 0.4011153, -0.2415066, 0.3204436, -0.6266803, 0.6426219
6: -0.2145945, 0.2954921, -0.1616421, 0.2305429, -0.4451374, 0.4571342
7: -0.2976781, 0.3024670, -0.2399483, 0.2344704, -0.5321485, 0.5424154
8: -0.2840665, 0.3684217, -0.2232168, 0.2880992, -0.5721657, 0.5916385
9: -0.2759777, 0.3555113, -0.2127742, 0.2770138, -0.5529915, 0.5682855

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6249942
time: 1.17 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6249942
time: 1.26 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.4234510, 1.0630963, 0.4824756, 1.0549092, -0.6314583, 0.5806207
1: -0.2715483, 0.2653010, -0.2371165, 0.2323934, -0.5039417, 0.5024175
2: -0.1883587, 0.3594979, -0.1579152, 0.3204614, -0.5088201, 0.5174131
3: -0.1977373, 0.2686198, -0.1726893, 0.2376928, -0.4354301, 0.4413091
4: -0.2855461, 0.2454836, -0.2536948, 0.2095042, -0.4950503, 0.4991784
5: -0.3062367, 0.4011153, -0.2706279, 0.3549570, -0.6611937, 0.6717432
6: -0.2145945, 0.2954921, -0.1845124, 0.2597626, -0.4743571, 0.4800045
7: -0.2976781, 0.3024670, -0.2651888, 0.2648947, -0.5625728, 0.5676558
8: -0.2840665, 0.3684217, -0.2503902, 0.3242351, -0.6083015, 0.6188118
9: -0.2759777, 0.3555113, -0.2409804, 0.3108306, -0.5868082, 0.5964917

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6249942
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6249942
time: 1.18 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.5177581, 1.0499723, 0.5683585, 1.0423906, -0.5246325, 0.4816138
1: -0.2165401, 0.2125497, -0.1871464, 0.1819962, -0.3985362, 0.3996961
2: -0.1398617, 0.2969218, -0.1169868, 0.2629058, -0.4027675, 0.4139085
3: -0.1576329, 0.2190433, -0.1342952, 0.1940155, -0.3516484, 0.3533385
4: -0.2344882, 0.1881642, -0.2058491, 0.1610488, -0.3955371, 0.3940132
5: -0.2491552, 0.3276372, -0.2144573, 0.2955474, -0.5447026, 0.5420945
6: -0.1665789, 0.2382173, -0.1456316, 0.2034019, -0.3699808, 0.3838489
7: -0.2458218, 0.2422382, -0.2191762, 0.2078591, -0.4536809, 0.4614144
8: -0.2300950, 0.2975902, -0.1997344, 0.2545342, -0.4846292, 0.4973246
9: -0.2198766, 0.2843478, -0.1888366, 0.2510769, -0.4709536, 0.4731844

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6227070, upper bound: 0.6231989
time: 1.44 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6236159, upper bound: 0.6232855
time: 1.14 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.5177581, 1.0499723, 0.5423588, 1.0462642, -0.5285060, 0.5076135
1: -0.2165401, 0.2125497, -0.2022496, 0.1976106, -0.4141507, 0.4147993
2: -0.1398617, 0.2969218, -0.1287405, 0.2801589, -0.4200205, 0.4256622
3: -0.1576329, 0.2190433, -0.1462867, 0.2065893, -0.3642223, 0.3653300
4: -0.2344882, 0.1881642, -0.2204303, 0.1749813, -0.4094695, 0.4085945
5: -0.2491552, 0.3276372, -0.2322859, 0.3119569, -0.5611121, 0.5599231
6: -0.1665789, 0.2382173, -0.1561843, 0.2212909, -0.3878698, 0.3944017
7: -0.2458218, 0.2422382, -0.2328674, 0.2253989, -0.4712207, 0.4751056
8: -0.2300950, 0.2975902, -0.2152120, 0.2766575, -0.5067524, 0.5128022
9: -0.2198766, 0.2843478, -0.2046142, 0.2681722, -0.4880489, 0.4889620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6227070, upper bound: 0.6231989
time: 1.75 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6236159, upper bound: 0.6232855
time: 1.19 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.4860784, 1.0544096, 0.5683585, 1.0423906, -0.5563122, 0.4860511
1: -0.2350148, 0.2303847, -0.1871464, 0.1819962, -0.4170110, 0.4175311
2: -0.1560569, 0.3180788, -0.1169868, 0.2629058, -0.4189627, 0.4350655
3: -0.1711604, 0.2358051, -0.1342952, 0.1940155, -0.3651758, 0.3701003
4: -0.2517508, 0.2073080, -0.2058491, 0.1610488, -0.4127997, 0.4131570
5: -0.2684543, 0.3521394, -0.2144573, 0.2955474, -0.5640017, 0.5665966
6: -0.1826762, 0.2575817, -0.1456316, 0.2034019, -0.3860781, 0.4032133
7: -0.2632056, 0.2626015, -0.2191762, 0.2078591, -0.4710647, 0.4817776
8: -0.2483346, 0.3215379, -0.1997344, 0.2545342, -0.5028689, 0.5212723
9: -0.2388442, 0.3081034, -0.1888366, 0.2510769, -0.4899211, 0.4969400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6175403, upper bound: 0.6189144
time: 1.17 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6192459
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.4860784, 1.0544096, 0.5423588, 1.0462642, -0.5601858, 0.5120509
1: -0.2350148, 0.2303847, -0.2022496, 0.1976106, -0.4326254, 0.4326343
2: -0.1560569, 0.3180788, -0.1287405, 0.2801589, -0.4362158, 0.4468192
3: -0.1711604, 0.2358051, -0.1462867, 0.2065893, -0.3777497, 0.3820918
4: -0.2517508, 0.2073080, -0.2204303, 0.1749813, -0.4267321, 0.4277383
5: -0.2684543, 0.3521394, -0.2322859, 0.3119569, -0.5804113, 0.5844253
6: -0.1826762, 0.2575817, -0.1561843, 0.2212909, -0.4039671, 0.4137660
7: -0.2632056, 0.2626015, -0.2328674, 0.2253989, -0.4886045, 0.4954689
8: -0.2483346, 0.3215379, -0.2152120, 0.2766575, -0.5249921, 0.5367499
9: -0.2388442, 0.3081034, -0.2046142, 0.2681722, -0.5070164, 0.5127176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6175403, upper bound: 0.6189144
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6192459
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.4034982, 1.0658638, 0.5288078, 1.0482831, -0.6447849, 0.5370561
1: -0.2831876, 0.2764251, -0.2101213, 0.2057486, -0.4889363, 0.4865464
2: -0.1986498, 0.3726937, -0.1348664, 0.2891510, -0.4878008, 0.5075601
3: -0.2062044, 0.2790744, -0.1525366, 0.2131427, -0.4193472, 0.4316110
4: -0.2963131, 0.2576461, -0.2280300, 0.1822429, -0.4785560, 0.4856761
5: -0.3182740, 0.4167187, -0.2415781, 0.3205094, -0.6387834, 0.6582968
6: -0.2247634, 0.3075702, -0.1616844, 0.2306146, -0.4553780, 0.4692546
7: -0.3086607, 0.3151678, -0.2400032, 0.2345406, -0.5432013, 0.5551710
8: -0.2954503, 0.3833583, -0.2232789, 0.2881878, -0.5836381, 0.6066372
9: -0.2878081, 0.3706150, -0.2128374, 0.2770824, -0.5648905, 0.5834523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6232855, upper bound: 0.6261892
time: 1.27 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6232855, upper bound: 0.6261892
time: 1.21 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.4034982, 1.0658638, 0.4823053, 1.0549331, -0.6514348, 0.5835585
1: -0.2831876, 0.2764251, -0.2372158, 0.2324882, -0.5156758, 0.5136409
2: -0.1986498, 0.3726937, -0.1580030, 0.3205740, -0.5192238, 0.5306967
3: -0.2062044, 0.2790744, -0.1727615, 0.2377819, -0.4439864, 0.4518359
4: -0.2963131, 0.2576461, -0.2537867, 0.2096079, -0.5059211, 0.5114329
5: -0.3182740, 0.4167187, -0.2707306, 0.3550899, -0.6733639, 0.6874492
6: -0.2247634, 0.3075702, -0.1845991, 0.2598657, -0.4846291, 0.4921692
7: -0.3086607, 0.3151678, -0.2652824, 0.2650031, -0.5736638, 0.5804502
8: -0.2954503, 0.3833583, -0.2504873, 0.3243624, -0.6198127, 0.6338456
9: -0.2878081, 0.3706150, -0.2410813, 0.3109595, -0.5987676, 0.6116962

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6232855, upper bound: 0.6261892
time: 1.18 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6232855, upper bound: 0.6261892
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.3803581, 1.0690734, 0.5288078, 1.0482831, -0.6679250, 0.5402657
1: -0.2966862, 0.2893263, -0.2101213, 0.2057486, -0.5024348, 0.4994476
2: -0.2105850, 0.3879978, -0.1348664, 0.2891510, -0.4997360, 0.5228642
3: -0.2160244, 0.2911992, -0.1525366, 0.2131427, -0.4291671, 0.4437358
4: -0.3088002, 0.2717518, -0.2280300, 0.1822429, -0.4910431, 0.4997817
5: -0.3322342, 0.4348146, -0.2415781, 0.3205094, -0.6527436, 0.6763927
6: -0.2365569, 0.3215777, -0.1616844, 0.2306146, -0.4671715, 0.4832621
7: -0.3213978, 0.3298978, -0.2400032, 0.2345406, -0.5559385, 0.5699010
8: -0.3086529, 0.4006813, -0.2232789, 0.2881878, -0.5968407, 0.6239602
9: -0.3015285, 0.3881317, -0.2128374, 0.2770824, -0.5786110, 0.6009691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6249899
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6249899
time: 1.20 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.3803581, 1.0690734, 0.4823053, 1.0549331, -0.6745750, 0.5867681
1: -0.2966862, 0.2893263, -0.2372158, 0.2324882, -0.5291744, 0.5265421
2: -0.2105850, 0.3879978, -0.1580030, 0.3205740, -0.5311590, 0.5460008
3: -0.2160244, 0.2911992, -0.1727615, 0.2377819, -0.4538063, 0.4639606
4: -0.3088002, 0.2717518, -0.2537867, 0.2096079, -0.5184081, 0.5255385
5: -0.3322342, 0.4348146, -0.2707306, 0.3550899, -0.6873241, 0.7055452
6: -0.2365569, 0.3215777, -0.1845991, 0.2598657, -0.4964226, 0.5061768
7: -0.3213978, 0.3298978, -0.2652824, 0.2650031, -0.5864010, 0.5951802
8: -0.3086529, 0.4006813, -0.2504873, 0.3243624, -0.6330153, 0.6511685
9: -0.3015285, 0.3881317, -0.2410813, 0.3109595, -0.6124880, 0.6292130

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6249899
time: 1.10 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6249899
time: 1.18 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.4605448, 1.0579513, 0.5682857, 1.0424014, -0.5818566, 0.4896656
1: -0.2499096, 0.2446203, -0.1871886, 0.1820398, -0.4319494, 0.4318089
2: -0.1692264, 0.3349654, -0.1170196, 0.2629541, -0.4321806, 0.4519851
3: -0.1819959, 0.2491838, -0.1343288, 0.1940506, -0.3760465, 0.3835126
4: -0.2655292, 0.2228723, -0.2058899, 0.1610878, -0.4266171, 0.4287623
5: -0.2838584, 0.3721070, -0.2145072, 0.2955934, -0.5794519, 0.5866142
6: -0.1956894, 0.2730380, -0.1456611, 0.2034520, -0.3991414, 0.4186991
7: -0.2772601, 0.2788546, -0.2192144, 0.2079083, -0.4851684, 0.4980690
8: -0.2629026, 0.3406527, -0.1997777, 0.2545961, -0.5174986, 0.5404303
9: -0.2539837, 0.3274317, -0.1888807, 0.2511247, -0.5051084, 0.5163124

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6186587, upper bound: 0.6196607
time: 1.17 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6196607, upper bound: 0.6196607
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.4605448, 1.0579513, 0.5422385, 1.0462822, -0.5857373, 0.5157129
1: -0.2499096, 0.2446203, -0.2023196, 0.1976828, -0.4475924, 0.4469399
2: -0.1692264, 0.3349654, -0.1287950, 0.2802388, -0.4494652, 0.4637604
3: -0.1819959, 0.2491838, -0.1463422, 0.2066477, -0.3886435, 0.3955260
4: -0.2655292, 0.2228723, -0.2204978, 0.1750459, -0.4405751, 0.4433702
5: -0.2838584, 0.3721070, -0.2323686, 0.3120329, -0.5958914, 0.6044756
6: -0.1956894, 0.2730380, -0.1562332, 0.2213738, -0.4170632, 0.4292712
7: -0.2772601, 0.2788546, -0.2329309, 0.2254803, -0.5027404, 0.5117854
8: -0.2629026, 0.3406527, -0.2152837, 0.2767600, -0.5396626, 0.5559364
9: -0.2539837, 0.3274317, -0.2046873, 0.2682515, -0.5222352, 0.5321190

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6186587, upper bound: 0.6196607
time: 1.17 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6196607, upper bound: 0.6196607
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.4399798, 1.0608034, 0.5682857, 1.0424014, -0.6024216, 0.4925177
1: -0.2619062, 0.2560859, -0.1871886, 0.1820398, -0.4439460, 0.4432745
2: -0.1798335, 0.3485664, -0.1170196, 0.2629541, -0.4427876, 0.4655860
3: -0.1907230, 0.2599593, -0.1343288, 0.1940506, -0.3847736, 0.3942881
4: -0.2766268, 0.2354082, -0.2058899, 0.1610878, -0.4377146, 0.4412981
5: -0.2962650, 0.3881893, -0.2145072, 0.2955934, -0.5918584, 0.6026964
6: -0.2061705, 0.2854867, -0.1456611, 0.2034520, -0.4096225, 0.4311478
7: -0.2885799, 0.2919455, -0.2192144, 0.2079083, -0.4964882, 0.5111600
8: -0.2746360, 0.3560478, -0.1997777, 0.2545961, -0.5292321, 0.5558255
9: -0.2661773, 0.3429992, -0.1888807, 0.2511247, -0.5173020, 0.5318799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6165653, upper bound: 0.6187838
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6181361, upper bound: 0.6192458
time: 1.12 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.4399798, 1.0608034, 0.5422385, 1.0462822, -0.6063024, 0.5185649
1: -0.2619062, 0.2560859, -0.2023196, 0.1976828, -0.4595891, 0.4584055
2: -0.1798335, 0.3485664, -0.1287950, 0.2802388, -0.4600723, 0.4773614
3: -0.1907230, 0.2599593, -0.1463422, 0.2066477, -0.3973706, 0.4063015
4: -0.2766268, 0.2354082, -0.2204978, 0.1750459, -0.4516726, 0.4559061
5: -0.2962650, 0.3881893, -0.2323686, 0.3120329, -0.6082979, 0.6205578
6: -0.2061705, 0.2854867, -0.1562332, 0.2213738, -0.4275443, 0.4417199
7: -0.2885799, 0.2919455, -0.2329309, 0.2254803, -0.5140602, 0.5248764
8: -0.2746360, 0.3560478, -0.2152837, 0.2767600, -0.5513960, 0.5713315
9: -0.2661773, 0.3429992, -0.2046873, 0.2682515, -0.5344288, 0.5476865

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6165653, upper bound: 0.6187838
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6181361, upper bound: 0.6192458
time: 1.14 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.4493957, 1.0594977, 0.4974344, 1.0528346, -0.6034389, 0.5620633
1: -0.2564134, 0.2508360, -0.2283904, 0.2240535, -0.4804669, 0.4792264
2: -0.1749770, 0.3423389, -0.1501997, 0.3105683, -0.4855453, 0.4925386
3: -0.1867271, 0.2550255, -0.1663413, 0.2298548, -0.4165819, 0.4213668
4: -0.2715454, 0.2296685, -0.2456228, 0.2003857, -0.4719311, 0.4752913
5: -0.2905845, 0.3808258, -0.2616034, 0.3432587, -0.6338432, 0.6424292
6: -0.2013715, 0.2797869, -0.1768885, 0.2507076, -0.4520791, 0.4566754
7: -0.2833969, 0.2859517, -0.2569548, 0.2553726, -0.5387695, 0.5429064
8: -0.2692636, 0.3489989, -0.2418555, 0.3130369, -0.5823005, 0.5908544
9: -0.2605941, 0.3358713, -0.2321109, 0.2995070, -0.5601012, 0.5679823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6288560, upper bound: 0.6268491
time: 1.23 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6268876, upper bound: 0.6268491
time: 1.16 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.4493957, 1.0594977, 0.4459375, 1.0599774, -0.6105817, 0.6135603
1: -0.2564134, 0.2508360, -0.2584308, 0.2527642, -0.5091776, 0.5092667
2: -0.1749770, 0.3423389, -0.1767606, 0.3446261, -0.5196031, 0.5190995
3: -0.1867271, 0.2550255, -0.1881947, 0.2568375, -0.4435646, 0.4432201
4: -0.2715454, 0.2296685, -0.2734118, 0.2317764, -0.5033219, 0.5030803
5: -0.2905845, 0.3808258, -0.2926707, 0.3835303, -0.6741147, 0.6734965
6: -0.2013715, 0.2797869, -0.2031340, 0.2818803, -0.4832519, 0.4829209
7: -0.2833969, 0.2859517, -0.2853006, 0.2881530, -0.5715499, 0.5712522
8: -0.2692636, 0.3489989, -0.2712367, 0.3515877, -0.6208514, 0.6202356
9: -0.2605941, 0.3358713, -0.2626446, 0.3384891, -0.5990832, 0.5985160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6288560, upper bound: 0.6268491
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6268876, upper bound: 0.6268491
time: 1.16 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.5199930, 1.0496200, 0.4974344, 1.0528346, -0.5328416, 0.5521857
1: -0.2152419, 0.2111335, -0.2283904, 0.2240535, -0.4392954, 0.4395239
2: -0.1388515, 0.2952418, -0.1501997, 0.3105683, -0.4494198, 0.4454415
3: -0.1566022, 0.2177125, -0.1663413, 0.2298548, -0.3864570, 0.3840538
4: -0.2331175, 0.1869666, -0.2456228, 0.2003857, -0.4335031, 0.4325894
5: -0.2476228, 0.3261575, -0.2616034, 0.3432587, -0.5908815, 0.5877609
6: -0.1654878, 0.2366798, -0.1768885, 0.2507076, -0.4161955, 0.4135683
7: -0.2446451, 0.2406213, -0.2569548, 0.2553726, -0.5000178, 0.4975760
8: -0.2286576, 0.2956887, -0.2418555, 0.3130369, -0.5416944, 0.5375443
9: -0.2183705, 0.2828783, -0.2321109, 0.2995070, -0.5178775, 0.5149893

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6218886, upper bound: 0.6238203
time: 1.12 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6206941, upper bound: 0.6238061
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.5199930, 1.0496200, 0.4459375, 1.0599774, -0.5399844, 0.6036826
1: -0.2152419, 0.2111335, -0.2584308, 0.2527642, -0.4680061, 0.4695643
2: -0.1388515, 0.2952418, -0.1767606, 0.3446261, -0.4834776, 0.4720024
3: -0.1566022, 0.2177125, -0.1881947, 0.2568375, -0.4134398, 0.4059072
4: -0.2331175, 0.1869666, -0.2734118, 0.2317764, -0.4648939, 0.4603784
5: -0.2476228, 0.3261575, -0.2926707, 0.3835303, -0.6311530, 0.6188282
6: -0.1654878, 0.2366798, -0.2031340, 0.2818803, -0.4473682, 0.4398138
7: -0.2446451, 0.2406213, -0.2853006, 0.2881530, -0.5327981, 0.5259218
8: -0.2286576, 0.2956887, -0.2712367, 0.3515877, -0.5802453, 0.5669255
9: -0.2183705, 0.2828783, -0.2626446, 0.3384891, -0.5568596, 0.5455229

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6218886, upper bound: 0.6238203
time: 1.17 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6206941, upper bound: 0.6238061
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.4840704, 1.0546880, 0.5528539, 1.0447005, -0.5606301, 0.5018340
1: -0.2361862, 0.2315043, -0.1961529, 0.1913076, -0.4274939, 0.4276572
2: -0.1570926, 0.3194066, -0.1239959, 0.2731945, -0.4302871, 0.4434024
3: -0.1720125, 0.2368571, -0.1414462, 0.2015136, -0.3735262, 0.3783033
4: -0.2528343, 0.2085319, -0.2145444, 0.1693572, -0.4221916, 0.4230763
5: -0.2696657, 0.3537096, -0.2250892, 0.3053330, -0.5749987, 0.5787988
6: -0.1836996, 0.2587972, -0.1519246, 0.2140697, -0.3977693, 0.4107218
7: -0.2643108, 0.2638797, -0.2273408, 0.2183186, -0.4826295, 0.4912204
8: -0.2494804, 0.3230411, -0.2089643, 0.2677271, -0.5172074, 0.5320054
9: -0.2400348, 0.3096234, -0.1982454, 0.2612715, -0.5013062, 0.5078687

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6200458, upper bound: 0.6197371
time: 1.23 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6217965, upper bound: 0.6202619
time: 1.15 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.4840704, 1.0546880, 0.5232522, 1.0491109, -0.5650405, 0.5314357
1: -0.2361862, 0.2315043, -0.2133485, 0.2090853, -0.4452715, 0.4448528
2: -0.1570926, 0.3194066, -0.1373780, 0.2928376, -0.4499303, 0.4567845
3: -0.1720125, 0.2368571, -0.1550990, 0.2158296, -0.3878421, 0.3919561
4: -0.2528343, 0.2085319, -0.2311457, 0.1852200, -0.4380543, 0.4396777
5: -0.2696657, 0.3537096, -0.2453878, 0.3240158, -0.5936815, 0.5990974
6: -0.1836996, 0.2587972, -0.1639393, 0.2344372, -0.4181368, 0.4227365
7: -0.2643108, 0.2638797, -0.2429287, 0.2382886, -0.5025994, 0.5068083
8: -0.2494804, 0.3230411, -0.2265862, 0.2929153, -0.5423956, 0.5496273
9: -0.2400348, 0.3096234, -0.2162088, 0.2807353, -0.5207700, 0.5258322

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6200458, upper bound: 0.6197371
time: 1.15 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6217965, upper bound: 0.6202619
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.4258893, 1.0627582, 0.5571777, 1.0440563, -0.6181670, 0.5055805
1: -0.2701258, 0.2639413, -0.1936412, 0.1887109, -0.4588368, 0.4575825
2: -0.1871010, 0.3578853, -0.1220413, 0.2703252, -0.4574262, 0.4799265
3: -0.1967024, 0.2673422, -0.1394521, 0.1994227, -0.3961251, 0.4067942
4: -0.2842302, 0.2439972, -0.2121194, 0.1670402, -0.4512705, 0.4561166
5: -0.3047656, 0.3992082, -0.2221242, 0.3026041, -0.6073697, 0.6213325
6: -0.2133518, 0.2940160, -0.1501697, 0.2110947, -0.4244465, 0.4441857
7: -0.2963358, 0.3009148, -0.2250638, 0.2154018, -0.5117376, 0.5259786
8: -0.2826752, 0.3665960, -0.2063903, 0.2640479, -0.5467231, 0.5729862
9: -0.2745318, 0.3536654, -0.1956215, 0.2584285, -0.5329604, 0.5492868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6187270, upper bound: 0.6197167
time: 1.20 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6206941, upper bound: 0.6202619
time: 1.19 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.4258893, 1.0627582, 0.5274499, 1.0484853, -0.6225960, 0.5353083
1: -0.2701258, 0.2639413, -0.2109102, 0.2065644, -0.4766902, 0.4748515
2: -0.1871010, 0.3578853, -0.1354804, 0.2900522, -0.4771532, 0.4933656
3: -0.1967024, 0.2673422, -0.1531629, 0.2137997, -0.4105021, 0.4205050
4: -0.2842302, 0.2439972, -0.2287916, 0.1829706, -0.4672009, 0.4727888
5: -0.3047656, 0.3992082, -0.2425095, 0.3213665, -0.6261320, 0.6417177
6: -0.2133518, 0.2940160, -0.1622356, 0.2315489, -0.4449006, 0.4562517
7: -0.2963358, 0.3009148, -0.2407183, 0.2354568, -0.5317925, 0.5416331
8: -0.2826752, 0.3665960, -0.2240873, 0.2893434, -0.5720186, 0.5906832
9: -0.2745318, 0.3536654, -0.2136615, 0.2779753, -0.5525071, 0.5673269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6187270, upper bound: 0.6197167
time: 1.12 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6206941, upper bound: 0.6202619
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.4072146, 1.0653484, 0.4970913, 1.0528822, -0.6456676, 0.5682571
1: -0.2810197, 0.2743532, -0.2285905, 0.2242448, -0.5052645, 0.5029438
2: -0.1967330, 0.3702359, -0.1503767, 0.3107952, -0.5075282, 0.5206127
3: -0.2046273, 0.2771271, -0.1664869, 0.2300346, -0.4346619, 0.4436139
4: -0.2943077, 0.2553808, -0.2458079, 0.2005949, -0.4949026, 0.5011887
5: -0.3160320, 0.4138122, -0.2618104, 0.3435271, -0.6595591, 0.6756226
6: -0.2228693, 0.3053206, -0.1770635, 0.2509154, -0.4737847, 0.4823840
7: -0.3066150, 0.3128022, -0.2571438, 0.2555911, -0.5622061, 0.5699459
8: -0.2933301, 0.3805762, -0.2420514, 0.3132936, -0.6066237, 0.6226276
9: -0.2856046, 0.3678018, -0.2323144, 0.2997668, -0.5853714, 0.6001163

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6289206, upper bound: 0.6289927
time: 1.19 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6289927, upper bound: 0.6289927
time: 1.14 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.4072146, 1.0653484, 0.4455451, 1.0600317, -0.6528171, 0.6198033
1: -0.2810197, 0.2743532, -0.2586597, 0.2529830, -0.5340027, 0.5330130
2: -0.1967330, 0.3702359, -0.1769630, 0.3448859, -0.5416188, 0.5471989
3: -0.2046273, 0.2771271, -0.1883612, 0.2570432, -0.4616706, 0.4654883
4: -0.2943077, 0.2553808, -0.2736234, 0.2320158, -0.5263235, 0.5290042
5: -0.3160320, 0.4138122, -0.2929076, 0.3838371, -0.6998691, 0.7067198
6: -0.2228693, 0.3053206, -0.2033342, 0.2821178, -0.5049870, 0.5086547
7: -0.3066150, 0.3128022, -0.2855166, 0.2884029, -0.5950179, 0.5983188
8: -0.2933301, 0.3805762, -0.2714608, 0.3518816, -0.6452117, 0.6520370
9: -0.2856046, 0.3678018, -0.2628775, 0.3387863, -0.6243908, 0.6306792

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6289206, upper bound: 0.6289927
time: 1.15 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6289927, upper bound: 0.6289927
time: 1.17 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.4697161, 1.0566792, 0.4970913, 1.0528822, -0.5831661, 0.5595880
1: -0.2445597, 0.2395070, -0.2285905, 0.2242448, -0.4688045, 0.4680976
2: -0.1644962, 0.3289000, -0.1503767, 0.3107952, -0.4752914, 0.4792768
3: -0.1781039, 0.2443784, -0.1664869, 0.2300346, -0.4081385, 0.4108653
4: -0.2605802, 0.2172819, -0.2458079, 0.2005949, -0.4611751, 0.4630898
5: -0.2783255, 0.3649350, -0.2618104, 0.3435271, -0.6218526, 0.6267453
6: -0.1910153, 0.2674863, -0.1770635, 0.2509154, -0.4419307, 0.4445497
7: -0.2722120, 0.2730168, -0.2571438, 0.2555911, -0.5278031, 0.5301605
8: -0.2576701, 0.3337868, -0.2420514, 0.3132936, -0.5709637, 0.5758381
9: -0.2485458, 0.3204893, -0.2323144, 0.2997668, -0.5483127, 0.5528037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213622, upper bound: 0.6238076
time: 1.19 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6202619, upper bound: 0.6237905
time: 1.17 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.4697161, 1.0566792, 0.4455451, 1.0600317, -0.5903155, 0.6111342
1: -0.2445597, 0.2395070, -0.2586597, 0.2529830, -0.4975427, 0.4981667
2: -0.1644962, 0.3289000, -0.1769630, 0.3448859, -0.5093821, 0.5058630
3: -0.1781039, 0.2443784, -0.1883612, 0.2570432, -0.4351471, 0.4327396
4: -0.2605802, 0.2172819, -0.2736234, 0.2320158, -0.4925960, 0.4909053
5: -0.2783255, 0.3649350, -0.2929076, 0.3838371, -0.6621626, 0.6578425
6: -0.1910153, 0.2674863, -0.2033342, 0.2821178, -0.4731330, 0.4708204
7: -0.2722120, 0.2730168, -0.2855166, 0.2884029, -0.5606149, 0.5585334
8: -0.2576701, 0.3337868, -0.2714608, 0.3518816, -0.6095517, 0.6052476
9: -0.2485458, 0.3204893, -0.2628775, 0.3387863, -0.5873321, 0.5833668

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6213622, upper bound: 0.6238076
time: 1.14 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6202619, upper bound: 0.6237905
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.3895567, 1.0677977, 0.5582044, 1.0439031, -0.6543464, 0.5095932
1: -0.2913203, 0.2841977, -0.1930447, 0.1880943, -0.4794146, 0.4772425
2: -0.2058405, 0.3819142, -0.1215770, 0.2696438, -0.4754844, 0.5034912
3: -0.2121207, 0.2863794, -0.1389784, 0.1989260, -0.4110467, 0.4253578
4: -0.3038363, 0.2661445, -0.2115436, 0.1664900, -0.4703263, 0.4776880
5: -0.3266848, 0.4276211, -0.2214200, 0.3019561, -0.6286409, 0.6490411
6: -0.2318688, 0.3160094, -0.1497529, 0.2103883, -0.4422571, 0.4657623
7: -0.3163345, 0.3240424, -0.2245231, 0.2147091, -0.5310436, 0.5485654
8: -0.3034047, 0.3937951, -0.2057790, 0.2631742, -0.5665790, 0.5995740
9: -0.2960743, 0.3811685, -0.1949984, 0.2577532, -0.5538275, 0.5761669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6146717, upper bound: 0.6170375
time: 1.14 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6167358, upper bound: 0.6178188
time: 1.17 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.3895567, 1.0677977, 0.5288730, 1.0482733, -0.6587166, 0.5389247
1: -0.2913203, 0.2841977, -0.2100835, 0.2057098, -0.4970301, 0.4942813
2: -0.2058405, 0.3819142, -0.1348371, 0.2891078, -0.4949484, 0.5167512
3: -0.2121207, 0.2863794, -0.1525067, 0.2131113, -0.4252320, 0.4388861
4: -0.3038363, 0.2661445, -0.2279935, 0.1822080, -0.4860443, 0.4941380
5: -0.3266848, 0.4276211, -0.2415336, 0.3204684, -0.6471532, 0.6691546
6: -0.2318688, 0.3160094, -0.1616580, 0.2305699, -0.4624387, 0.4776674
7: -0.3163345, 0.3240424, -0.2399690, 0.2344967, -0.5508312, 0.5640113
8: -0.3034047, 0.3937951, -0.2232402, 0.2881326, -0.5915374, 0.6170353
9: -0.2960743, 0.3811685, -0.2127980, 0.2770396, -0.5731139, 0.5939665

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=48, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6146717, upper bound: 0.6170375
time: 1.12 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6167358, upper bound: 0.6178188
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.4282873, 1.0624255, 0.5495731, 1.0451893, -0.6169020, 0.5128524
1: -0.2687270, 0.2626047, -0.1980588, 0.1932779, -0.4620049, 0.4606635
2: -0.1858643, 0.3562993, -0.1254791, 0.2753715, -0.4612358, 0.4817784
3: -0.1956849, 0.2660857, -0.1429594, 0.2031004, -0.3987853, 0.4090451
4: -0.2829364, 0.2425356, -0.2163843, 0.1711153, -0.4540517, 0.4589199
5: -0.3033190, 0.3973331, -0.2273389, 0.3074038, -0.6107228, 0.6246719
6: -0.2121295, 0.2925645, -0.1532562, 0.2163271, -0.4284567, 0.4458207
7: -0.2950160, 0.2993884, -0.2290684, 0.2205320, -0.5155480, 0.5284567
8: -0.2813072, 0.3648011, -0.2109173, 0.2705188, -0.5518260, 0.5757184
9: -0.2731100, 0.3518502, -0.2002363, 0.2634285, -0.5365386, 0.5520864

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6165653, upper bound: 0.6181154
time: 1.17 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6181361, upper bound: 0.6181361
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.4282873, 1.0624255, 0.5201602, 1.0495936, -0.6213063, 0.5422653
1: -0.2687270, 0.2626047, -0.2151447, 0.2110277, -0.4797547, 0.4777494
2: -0.1858643, 0.3562993, -0.1387758, 0.2951162, -0.4809805, 0.4950751
3: -0.1956849, 0.2660857, -0.1565251, 0.2176129, -0.4132978, 0.4226109
4: -0.2829364, 0.2425356, -0.2330150, 0.1868770, -0.4698133, 0.4755506
5: -0.3033190, 0.3973331, -0.2475081, 0.3260469, -0.6293659, 0.6448411
6: -0.2121295, 0.2925645, -0.1654062, 0.2365647, -0.4486942, 0.4579707
7: -0.2950160, 0.2993884, -0.2445571, 0.2405002, -0.5355162, 0.5439455
8: -0.2813072, 0.3648011, -0.2285500, 0.2955462, -0.5768534, 0.5933511
9: -0.2731100, 0.3518502, -0.2182578, 0.2827684, -0.5558784, 0.5701079

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6165653, upper bound: 0.6181154
time: 1.24 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6181361, upper bound: 0.6181361
time: 1.03 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.21 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6198216, upper bound: 0.6215045
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6198216, upper bound: 0.6215045
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6198216, upper bound: 0.6225501
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6198216, upper bound: 0.6225501
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6200984
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6200984
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6213442
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6213442
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6207517, upper bound: 0.6222330
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6220421, upper bound: 0.6229761
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6207517, upper bound: 0.6212997
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6220421, upper bound: 0.6216394
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6187270, upper bound: 0.6207624
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6206941, upper bound: 0.6220167
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6187270, upper bound: 0.6201649
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6206941, upper bound: 0.6212123
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6196607, upper bound: 0.6214766
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6196607, upper bound: 0.6214766
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6196607, upper bound: 0.6225322
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6196607, upper bound: 0.6225322
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6181361, upper bound: 0.6200550
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6181361, upper bound: 0.6200550
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6181361, upper bound: 0.6213309
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6181361, upper bound: 0.6213309
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6205192, upper bound: 0.6222430
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6216394, upper bound: 0.6229761
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6205192, upper bound: 0.6213032
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6216394, upper bound: 0.6216394
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6187256, upper bound: 0.6207901
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6202619, upper bound: 0.6220167
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6187256, upper bound: 0.6201923
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6202619, upper bound: 0.6212118
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6252939, upper bound: 0.6223945
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6236611, upper bound: 0.6223945
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6255694, upper bound: 0.6236558
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6236611, upper bound: 0.6236558
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6182497, upper bound: 0.6185786
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6171597, upper bound: 0.6185352
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6184171, upper bound: 0.6201433
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6171597, upper bound: 0.6201143
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6155709, upper bound: 0.6152664
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6171691, upper bound: 0.6152664
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6173258, upper bound: 0.6170159
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6183778, upper bound: 0.6170159
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6148784, upper bound: 0.6152647
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6164195, upper bound: 0.6152664
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6162086, upper bound: 0.6170155
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6171597, upper bound: 0.6170159
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6252907, upper bound: 0.6223945
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6236558, upper bound: 0.6223945
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6255477, upper bound: 0.6236558
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6236558, upper bound: 0.6236558
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6180263, upper bound: 0.6185691
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6170159, upper bound: 0.6185263
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6181335, upper bound: 0.6201294
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6170159, upper bound: 0.6201044
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6150744, upper bound: 0.6152567
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6171072, upper bound: 0.6152664
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6164850, upper bound: 0.6170137
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6181180, upper bound: 0.6170159
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6143194, upper bound: 0.6152486
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6163938, upper bound: 0.6152664
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6151921, upper bound: 0.6170130
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6170159, upper bound: 0.6170159
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6236159, upper bound: 0.6261992
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6236159, upper bound: 0.6261992
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6236159, upper bound: 0.6261992
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6236159, upper bound: 0.6261992
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6249942
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6249942
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6249942
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6249942
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6227070, upper bound: 0.6231989
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6236159, upper bound: 0.6232855
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6227070, upper bound: 0.6231989
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6236159, upper bound: 0.6232855
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6175403, upper bound: 0.6189144
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6192459
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6175403, upper bound: 0.6189144
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6192459
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6232855, upper bound: 0.6261892
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6232855, upper bound: 0.6261892
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6232855, upper bound: 0.6261892
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6232855, upper bound: 0.6261892
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6249899
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6249899
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6249899
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6249899
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6186587, upper bound: 0.6196607
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6196607, upper bound: 0.6196607
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6186587, upper bound: 0.6196607
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6196607, upper bound: 0.6196607
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6165653, upper bound: 0.6187838
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6181361, upper bound: 0.6192458
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6165653, upper bound: 0.6187838
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6181361, upper bound: 0.6192458
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6288560, upper bound: 0.6268491
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6268876, upper bound: 0.6268491
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6288560, upper bound: 0.6268491
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6268876, upper bound: 0.6268491
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6218886, upper bound: 0.6238203
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6206941, upper bound: 0.6238061
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6218886, upper bound: 0.6238203
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6206941, upper bound: 0.6238061
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6200458, upper bound: 0.6197371
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6217965, upper bound: 0.6202619
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6200458, upper bound: 0.6197371
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6217965, upper bound: 0.6202619
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6187270, upper bound: 0.6197167
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6206941, upper bound: 0.6202619
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6187270, upper bound: 0.6197167
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6206941, upper bound: 0.6202619
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6289206, upper bound: 0.6289927
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6289927, upper bound: 0.6289927
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6289206, upper bound: 0.6289927
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6289927, upper bound: 0.6289927
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6213622, upper bound: 0.6238076
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6202619, upper bound: 0.6237905
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6213622, upper bound: 0.6238076
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6202619, upper bound: 0.6237905
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6146717, upper bound: 0.6170375
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6167358, upper bound: 0.6178188
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6146717, upper bound: 0.6170375
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6167358, upper bound: 0.6178188
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6165653, upper bound: 0.6181154
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6181361, upper bound: 0.6181361
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6165653, upper bound: 0.6181154
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -0.6181361, upper bound: 0.6181361

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.5750464, 1.0413940, 0.5616872, 1.0433842, -0.4683378, 0.4797068
1: -0.1832612, 0.1779797, -0.1910217, 0.1860027, -0.3692639, 0.3690014
2: -0.1139633, 0.2584677, -0.1200027, 0.2673327, -0.3812961, 0.3784704
3: -0.1312107, 0.1907810, -0.1373721, 0.1972418, -0.3284525, 0.3281531
4: -0.2020982, 0.1574648, -0.2095905, 0.1646237, -0.3667219, 0.3670554
5: -0.2098711, 0.2913265, -0.2190318, 0.2997580, -0.5096291, 0.5103583
6: -0.1429170, 0.1988002, -0.1483393, 0.2079921, -0.3509092, 0.3471395
7: -0.2156543, 0.2033472, -0.2226892, 0.2123595, -0.4280139, 0.4260363
8: -0.1957530, 0.2488433, -0.2037058, 0.2602109, -0.4559640, 0.4525491
9: -0.1847780, 0.2466793, -0.1928850, 0.2554634, -0.4402414, 0.4395642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=41, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=46, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6183543, upper bound: 0.6202863
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6183543, upper bound: 0.6194745
time: 1.28 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.6144685, 1.0355206, 0.5616872, 1.0433842, -0.4289157, 0.4738333
1: -0.1603609, 0.1543040, -0.1910217, 0.1860027, -0.3463636, 0.3453257
2: -0.0961418, 0.2323077, -0.1200027, 0.2673327, -0.3634745, 0.3523104
3: -0.1130284, 0.1717159, -0.1373721, 0.1972418, -0.3102703, 0.3090880
4: -0.1799893, 0.1363396, -0.2095905, 0.1646237, -0.3446130, 0.3459302
5: -0.1828383, 0.2664455, -0.2190318, 0.2997580, -0.4825963, 0.4854773
6: -0.1269164, 0.1716759, -0.1483393, 0.2079921, -0.3349085, 0.3200152
7: -0.1948949, 0.1767523, -0.2226892, 0.2123595, -0.4072544, 0.3994415
8: -0.1722849, 0.2152990, -0.2037058, 0.2602109, -0.4324959, 0.4190047
9: -0.1608551, 0.2207582, -0.1928850, 0.2554634, -0.4163185, 0.4136432

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=40, inp2_unstable=41, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=46, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6183543, upper bound: 0.6202863
time: 1.28 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6183543, upper bound: 0.6194745
time: 13.75 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 16.91 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 16.91
Output dim: 0, lower bound: -0.6183543, upper bound: 0.6202863
IS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 16.91
Output dim: 0, lower bound: -0.6183543, upper bound: 0.6194745
IS_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 16.91
Output dim: 0, lower bound: -0.6183543, upper bound: 0.6202863
IS_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 16.91
Output dim: 0, lower bound: -0.6183543, upper bound: 0.6194745
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6198216, upper bound: 0.6225501
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6198216, upper bound: 0.6225501
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6200984
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6200984
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6213442
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6213442
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6207517, upper bound: 0.6222330
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6220421, upper bound: 0.6229761
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6207517, upper bound: 0.6212997
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6220421, upper bound: 0.6216394
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6187270, upper bound: 0.6207624
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6206941, upper bound: 0.6220167
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6187270, upper bound: 0.6201649
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6206941, upper bound: 0.6212123
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6196607, upper bound: 0.6214766
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6196607, upper bound: 0.6214766
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6196607, upper bound: 0.6225322
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6196607, upper bound: 0.6225322
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6181361, upper bound: 0.6200550
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6181361, upper bound: 0.6200550
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6181361, upper bound: 0.6213309
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6181361, upper bound: 0.6213309
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6205192, upper bound: 0.6222430
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6216394, upper bound: 0.6229761
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6205192, upper bound: 0.6213032
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6216394, upper bound: 0.6216394
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6187256, upper bound: 0.6207901
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6202619, upper bound: 0.6220167
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6187256, upper bound: 0.6201923
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6202619, upper bound: 0.6212118
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6252939, upper bound: 0.6223945
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6236611, upper bound: 0.6223945
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6255694, upper bound: 0.6236558
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6236611, upper bound: 0.6236558
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6182497, upper bound: 0.6185786
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6171597, upper bound: 0.6185352
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6184171, upper bound: 0.6201433
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6171597, upper bound: 0.6201143
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6155709, upper bound: 0.6152664
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6171691, upper bound: 0.6152664
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6173258, upper bound: 0.6170159
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6183778, upper bound: 0.6170159
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6148784, upper bound: 0.6152647
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6164195, upper bound: 0.6152664
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6162086, upper bound: 0.6170155
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6171597, upper bound: 0.6170159
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6252907, upper bound: 0.6223945
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6236558, upper bound: 0.6223945
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6255477, upper bound: 0.6236558
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6236558, upper bound: 0.6236558
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6180263, upper bound: 0.6185691
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6170159, upper bound: 0.6185263
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6181335, upper bound: 0.6201294
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6170159, upper bound: 0.6201044
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6150744, upper bound: 0.6152567
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6171072, upper bound: 0.6152664
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6164850, upper bound: 0.6170137
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6181180, upper bound: 0.6170159
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6143194, upper bound: 0.6152486
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6163938, upper bound: 0.6152664
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6151921, upper bound: 0.6170130
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6170159, upper bound: 0.6170159
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6236159, upper bound: 0.6261992
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6236159, upper bound: 0.6261992
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6236159, upper bound: 0.6261992
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6236159, upper bound: 0.6261992
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6249942
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6249942
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6249942
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6219555, upper bound: 0.6249942
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6227070, upper bound: 0.6231989
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6236159, upper bound: 0.6232855
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6227070, upper bound: 0.6231989
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6236159, upper bound: 0.6232855
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6175403, upper bound: 0.6189144
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6192459
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6175403, upper bound: 0.6189144
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6184210, upper bound: 0.6192459
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6232855, upper bound: 0.6261892
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6232855, upper bound: 0.6261892
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6232855, upper bound: 0.6261892
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6232855, upper bound: 0.6261892
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6249899
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6249899
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6249899
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6213716, upper bound: 0.6249899
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6186587, upper bound: 0.6196607
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6196607, upper bound: 0.6196607
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6186587, upper bound: 0.6196607
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6196607, upper bound: 0.6196607
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6165653, upper bound: 0.6187838
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6181361, upper bound: 0.6192458
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6165653, upper bound: 0.6187838
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6181361, upper bound: 0.6192458
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6288560, upper bound: 0.6268491
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6268876, upper bound: 0.6268491
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6288560, upper bound: 0.6268491
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6268876, upper bound: 0.6268491
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6218886, upper bound: 0.6238203
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6206941, upper bound: 0.6238061
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6218886, upper bound: 0.6238203
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6206941, upper bound: 0.6238061
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6200458, upper bound: 0.6197371
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6217965, upper bound: 0.6202619
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6200458, upper bound: 0.6197371
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6217965, upper bound: 0.6202619
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6187270, upper bound: 0.6197167
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6206941, upper bound: 0.6202619
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6187270, upper bound: 0.6197167
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6206941, upper bound: 0.6202619
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6289206, upper bound: 0.6289927
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6289927, upper bound: 0.6289927
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6289206, upper bound: 0.6289927
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6289927, upper bound: 0.6289927
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6213622, upper bound: 0.6238076
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6202619, upper bound: 0.6237905
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6213622, upper bound: 0.6238076
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6202619, upper bound: 0.6237905
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6146717, upper bound: 0.6170375
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6167358, upper bound: 0.6178188
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6146717, upper bound: 0.6170375
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6167358, upper bound: 0.6178188
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6165653, upper bound: 0.6181154
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6181361, upper bound: 0.6181361
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6165653, upper bound: 0.6181154
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.91
Output dim: 0, lower bound: -0.6181361, upper bound: 0.6181361

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 4.43 + 602.34 = 606.77 seconds
