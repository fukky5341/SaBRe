## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 460.407499041
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-256.7159119, 203.7015533, -256.7159119, 203.7015533, -460.4174805, 460.4174805)
1: (-215.5436707, 181.1157990, -215.5436707, 181.1157990, -396.6594543, 396.6594543)
2: (-282.8133240, 182.7726288, -282.8133240, 182.7726288, -465.5859375, 465.5859375)
3: (-301.2575073, 158.7781830, -301.2575073, 158.7781830, -460.0357056, 460.0357056)
4: (-276.2066956, 210.6250763, -276.2066956, 210.6250763, -486.8317566, 486.8317566)
5: (-246.9300537, 191.3716736, -246.9300537, 191.3716736, -438.3016968, 438.3016968)
6: (-236.3738251, 227.5385132, -236.3738251, 227.5385132, -463.9123535, 463.9123535)
7: (-257.5447693, 215.6144562, -257.5447693, 215.6144562, -473.1591797, 473.1591797)
8: (-309.6375427, 210.9121857, -309.6375427, 210.9121857, -520.5497437, 520.5497437)
9: (-234.0735016, 229.8993225, -234.0735016, 229.8993225, -463.9727783, 463.9727783)

## BASE Result
execution time: IAR + LP analysis = 1.24 + 13.02 = 14.26 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -460.4076444, upper bound: 460.4076444


# Binary Search by BASE starts (time budget: 2685.74 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=463.9727783203125
rel_dist={9: [-460.40761108255293, 460.4076110701256]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=463.9727783203125
rel_dist={9: [-460.40755870218874, 460.4075586754011]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=463.9727783203125
rel_dist={9: [-460.40746283482497, 460.4074627920745]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.UNKNOWN, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=463.9727783203125
rel_dist={9: [-460.4075190791105, 460.4075190850241]}

## Binary Search Result
Binary search time: 71.47 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.00390625


# Individual Split (IS_dual_ind) starts
Time budget: 2614.27 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076029, upper bound: 460.4076060
time: 9.71 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4075979, upper bound: 460.4075978
time: 12.55 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 22.37 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 22.37
Output dim: 9, lower bound: -460.4076029, upper bound: 460.4076060
IS_A2, status: Status.UNKNOWN, split count: 1, time: 22.37
Output dim: 9, lower bound: -460.4075979, upper bound: 460.4075978

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -249.9490967, 198.3380127, -256.6192627, 203.6248322, -453.5739136, 454.9572754
1: -209.8780212, 176.3652649, -215.4626923, 181.0477448, -390.9257812, 391.8279114
2: -275.3728638, 177.9798737, -282.7068481, 182.7039948, -458.0768433, 460.6867065
3: -293.3232117, 154.6322174, -301.1435242, 158.7184448, -452.0416565, 455.7756958
4: -268.9162598, 205.0838013, -276.1024170, 210.5459442, -479.4621582, 481.1862183
5: -240.4125671, 186.3449554, -246.8371124, 191.2996826, -431.7122498, 433.1820679
6: -230.1435394, 221.5507507, -236.2847290, 227.4526978, -457.5962524, 457.8354187
7: -250.7793427, 209.9461670, -257.4479980, 215.5330505, -466.3123779, 467.3941650
8: -301.4733582, 205.3209381, -309.5207825, 210.8325348, -512.3058472, 514.8417358
9: -227.9011993, 223.8585358, -233.9848785, 229.8124237, -457.7135925, 457.8434143

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4075016, upper bound: 460.4074632
time: 9.51 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074249, upper bound: 460.4074253
time: 9.91 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -254.1189270, 201.6398621, -256.7159119, 203.7015533, -457.8204956, 458.3557739
1: -213.3681488, 179.2900391, -215.5436707, 181.1157990, -394.4839478, 394.8337097
2: -279.9508972, 180.9277039, -282.8133240, 182.7726288, -462.7235107, 463.7410278
3: -298.1987915, 157.1773987, -301.2575073, 158.7781830, -456.9769897, 458.4349060
4: -273.4079895, 208.4995575, -276.2066956, 210.6250763, -484.0330200, 484.7062378
5: -244.4365845, 189.4364014, -246.9300537, 191.3716736, -435.8082581, 436.3664246
6: -233.9858398, 225.2335358, -236.3738251, 227.5385132, -461.5243530, 461.6073608
7: -254.9468689, 213.4310913, -257.5447693, 215.6144562, -470.5613403, 470.9757996
8: -306.4979553, 208.7694855, -309.6375427, 210.9121857, -517.4101562, 518.4070435
9: -231.6973724, 227.5678711, -234.0735016, 229.8993225, -461.5966492, 461.6413574

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4075979, upper bound: 460.4075978
time: 9.50 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4075979, upper bound: 460.4075978
time: 10.40 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 21.16 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 21.16
Output dim: 9, lower bound: -460.4075016, upper bound: 460.4074632
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 21.16
Output dim: 9, lower bound: -460.4074249, upper bound: 460.4074253
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 21.16
Output dim: 9, lower bound: -460.4075979, upper bound: 460.4075978
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 21.16
Output dim: 9, lower bound: -460.4075979, upper bound: 460.4075978

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -249.9490967, 198.3380127, -254.4666901, 201.9212646, -451.8703003, 452.8046875
1: -209.8780212, 176.3652649, -213.6626892, 179.5406494, -389.4186707, 390.0279236
2: -275.3728638, 177.9798737, -280.3357849, 181.1753082, -456.5481262, 458.3156433
3: -293.3232117, 154.6322174, -298.6246338, 157.4039764, -450.7271729, 453.2568054
4: -268.9162598, 205.0838013, -273.7851868, 208.7767639, -477.6930237, 478.8689880
5: -240.4125671, 186.3449554, -244.7782745, 189.6980286, -430.1105957, 431.1231995
6: -230.1435394, 221.5507507, -234.3134155, 225.5469360, -455.6904907, 455.8641663
7: -250.7793427, 209.9461670, -255.2909241, 213.7270355, -464.5063782, 465.2370911
8: -301.4733582, 205.3209381, -306.9296875, 209.0633087, -510.5365906, 512.2506104
9: -227.9011993, 223.8585358, -232.0151215, 227.8810883, -455.7822266, 455.8736572

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074249, upper bound: 460.4074254
time: 10.05 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074249, upper bound: 460.4074254
time: 9.67 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -254.1189270, 201.6398621, -249.9490967, 198.3380127, -452.4569397, 451.5889587
1: -213.3681488, 179.2900391, -209.8780212, 176.3652649, -389.7333374, 389.1680603
2: -279.9508972, 180.9277039, -275.3728638, 177.9798737, -457.9307556, 456.3005676
3: -298.1987915, 157.1773987, -293.3232117, 154.6322174, -452.8309631, 450.5006104
4: -273.4079895, 208.4995575, -268.9162598, 205.0838013, -478.4917908, 477.4158325
5: -244.4365845, 189.4364014, -240.4125671, 186.3449554, -430.7815552, 429.8489685
6: -233.9858398, 225.2335358, -230.1435394, 221.5507507, -455.5365906, 455.3770752
7: -254.9468689, 213.4310913, -250.7793427, 209.9461670, -464.8930359, 464.2104187
8: -306.4979553, 208.7694855, -301.4733582, 205.3209381, -511.8189087, 510.2428589
9: -231.6973724, 227.5678711, -227.9011993, 223.8585358, -455.5559082, 455.4690552

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4074621, upper bound: 460.4075013
time: 10.23 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074236, upper bound: 460.4074236
time: 11.75 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -254.1189270, 201.6398621, -254.1189270, 201.6398621, -455.7587891, 455.7587891
1: -213.3681488, 179.2900391, -213.3681488, 179.2900391, -392.6582031, 392.6582031
2: -279.9508972, 180.9277039, -279.9508972, 180.9277039, -460.8786011, 460.8786011
3: -298.1987915, 157.1773987, -298.1987915, 157.1773987, -455.3761902, 455.3761902
4: -273.4079895, 208.4995575, -273.4079895, 208.4995575, -481.9075317, 481.9075317
5: -244.4365845, 189.4364014, -244.4365845, 189.4364014, -433.8729858, 433.8729858
6: -233.9858398, 225.2335358, -233.9858398, 225.2335358, -459.2193604, 459.2193604
7: -254.9468689, 213.4310913, -254.9468689, 213.4310913, -468.3779602, 468.3779602
8: -306.4979553, 208.7694855, -306.4979553, 208.7694855, -515.2674561, 515.2674561
9: -231.6973724, 227.5678711, -231.6973724, 227.5678711, -459.2652588, 459.2652588

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4074621, upper bound: 460.4075012
time: 10.54 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074236, upper bound: 460.4074236
time: 11.38 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 23.20 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 23.20
Output dim: 9, lower bound: -460.4074249, upper bound: 460.4074254
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 23.20
Output dim: 9, lower bound: -460.4074249, upper bound: 460.4074254
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 23.20
Output dim: 9, lower bound: -460.4074621, upper bound: 460.4075013
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 23.20
Output dim: 9, lower bound: -460.4074236, upper bound: 460.4074236
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 23.20
Output dim: 9, lower bound: -460.4074621, upper bound: 460.4075012
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 23.20
Output dim: 9, lower bound: -460.4074236, upper bound: 460.4074236

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -251.9547577, 199.9270782, -249.9490967, 198.3380127, -450.2927856, 449.8761597
1: -211.5583649, 177.7747040, -209.8780212, 176.3652649, -387.9236145, 387.6527100
2: -277.5670166, 179.3906555, -275.3728638, 177.9798737, -455.5468750, 454.7635193
3: -295.6665039, 155.8557281, -293.3232117, 154.6322174, -450.2987061, 449.1789551
4: -271.0782776, 206.7208862, -268.9162598, 205.0838013, -476.1620789, 475.6371460
5: -242.3665619, 187.8260651, -240.4125671, 186.3449554, -428.7114868, 428.2386475
6: -232.0039215, 223.3173828, -230.1435394, 221.5507507, -453.5546265, 453.4609375
7: -252.7781677, 211.6152496, -250.7793427, 209.9461670, -462.7243347, 462.3945923
8: -303.8927612, 206.9906006, -301.4733582, 205.3209381, -509.2136841, 508.4639587
9: -229.7169495, 225.6260376, -227.9011993, 223.8585358, -453.5754089, 453.5271912

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074254, upper bound: 460.4074249
time: 10.51 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074254, upper bound: 460.4074249
time: 11.31 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -251.9547577, 199.9270782, -254.1189270, 201.6398621, -453.5946045, 454.0460205
1: -211.5583649, 177.7747040, -213.3681488, 179.2900391, -390.8483887, 391.1428223
2: -277.5670166, 179.3906555, -279.9508972, 180.9277039, -458.4947205, 459.3415527
3: -295.6665039, 155.8557281, -298.1987915, 157.1773987, -452.8439026, 454.0545044
4: -271.0782776, 206.7208862, -273.4079895, 208.4995575, -479.5778198, 480.1288757
5: -242.3665619, 187.8260651, -244.4365845, 189.4364014, -431.8029480, 432.2626343
6: -232.0039215, 223.3173828, -233.9858398, 225.2335358, -457.2374573, 457.3032227
7: -252.7781677, 211.6152496, -254.9468689, 213.4310913, -466.2092285, 466.5621338
8: -303.8927612, 206.9906006, -306.4979553, 208.7694855, -512.6621704, 513.4885254
9: -229.7169495, 225.6260376, -231.6973724, 227.5678711, -457.2847595, 457.3234253

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074236, upper bound: 460.4074236
time: 10.42 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074236, upper bound: 460.4074236
time: 16.71 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 28.40 seconds
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 28.40
Output dim: 9, lower bound: -460.4074254, upper bound: 460.4074249
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 28.40
Output dim: 9, lower bound: -460.4074254, upper bound: 460.4074249
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 28.40
Output dim: 9, lower bound: -460.4074236, upper bound: 460.4074236
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 28.40
Output dim: 9, lower bound: -460.4074236, upper bound: 460.4074236
Binary search (step 0): status=Status.VERIFIED, k_low=2, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=463.9727783203125
rel_dist={9: [-460.4076172501469, 460.40761719415866]}

## Binary search (step 1) starts
Candidate k: 10, corresponding eps: 0.0390625


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076213, upper bound: 460.4076249
time: 10.17 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076157, upper bound: 460.4076157
time: 9.63 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 19.92 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 19.92
Output dim: 9, lower bound: -460.4076213, upper bound: 460.4076249
IS_A2, status: Status.UNKNOWN, split count: 1, time: 19.92
Output dim: 9, lower bound: -460.4076157, upper bound: 460.4076157

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -249.9490967, 198.3380127, -256.7159119, 203.7015533, -453.6506348, 455.0539246
1: -209.8780212, 176.3652649, -215.5436707, 181.1157990, -390.9938354, 391.9088745
2: -275.3728638, 177.9798737, -282.8133240, 182.7726288, -458.1454773, 460.7931824
3: -293.3232117, 154.6322174, -301.2575073, 158.7781830, -452.1013794, 455.8896790
4: -268.9162598, 205.0838013, -276.2066956, 210.6250763, -479.5413208, 481.2904968
5: -240.4125671, 186.3449554, -246.9300537, 191.3716736, -431.7842407, 433.2749634
6: -230.1435394, 221.5507507, -236.3738251, 227.5385132, -457.6820679, 457.9245605
7: -250.7793427, 209.9461670, -257.5447693, 215.6144562, -466.3937988, 467.4909058
8: -301.4733582, 205.3209381, -309.6375427, 210.9121857, -512.3855591, 514.9584961
9: -227.9011993, 223.8585358, -234.0735016, 229.8993225, -457.8004456, 457.9320374

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076157, upper bound: 460.4076157
time: 9.69 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076157, upper bound: 460.4076157
time: 10.29 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -254.1189270, 201.6398621, -256.7159119, 203.7015533, -457.8204956, 458.3557739
1: -213.3681488, 179.2900391, -215.5436707, 181.1157990, -394.4839478, 394.8337097
2: -279.9508972, 180.9277039, -282.8133240, 182.7726288, -462.7235107, 463.7410278
3: -298.1987915, 157.1773987, -301.2575073, 158.7781830, -456.9769897, 458.4349060
4: -273.4079895, 208.4995575, -276.2066956, 210.6250763, -484.0330200, 484.7062378
5: -244.4365845, 189.4364014, -246.9300537, 191.3716736, -435.8082581, 436.3664246
6: -233.9858398, 225.2335358, -236.3738251, 227.5385132, -461.5243530, 461.6073608
7: -254.9468689, 213.4310913, -257.5447693, 215.6144562, -470.5613403, 470.9757996
8: -306.4979553, 208.7694855, -309.6375427, 210.9121857, -517.4101562, 518.4070435
9: -231.6973724, 227.5678711, -234.0735016, 229.8993225, -461.5966492, 461.6413574

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076157, upper bound: 460.4076157
time: 12.51 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076157, upper bound: 460.4076157
time: 8.62 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.42 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 22.42
Output dim: 9, lower bound: -460.4076157, upper bound: 460.4076157
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 22.42
Output dim: 9, lower bound: -460.4076157, upper bound: 460.4076157
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 22.42
Output dim: 9, lower bound: -460.4076157, upper bound: 460.4076157
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 22.42
Output dim: 9, lower bound: -460.4076157, upper bound: 460.4076157

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -249.9490967, 198.3380127, -249.9490967, 198.3380127, -448.2871094, 448.2871094
1: -209.8780212, 176.3652649, -209.8780212, 176.3652649, -386.2432556, 386.2432556
2: -275.3728638, 177.9798737, -275.3728638, 177.9798737, -453.3527222, 453.3527222
3: -293.3232117, 154.6322174, -293.3232117, 154.6322174, -447.9553833, 447.9553833
4: -268.9162598, 205.0838013, -268.9162598, 205.0838013, -474.0000610, 474.0000610
5: -240.4125671, 186.3449554, -240.4125671, 186.3449554, -426.7575073, 426.7575073
6: -230.1435394, 221.5507507, -230.1435394, 221.5507507, -451.6942749, 451.6942749
7: -250.7793427, 209.9461670, -250.7793427, 209.9461670, -460.7255249, 460.7255249
8: -301.4733582, 205.3209381, -301.4733582, 205.3209381, -506.7943115, 506.7943115
9: -227.9011993, 223.8585358, -227.9011993, 223.8585358, -451.7597351, 451.7597351

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4074916, upper bound: 460.4075449
time: 9.98 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074379, upper bound: 460.4074380
time: 9.77 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -249.9490967, 198.3380127, -254.1189270, 201.6398621, -451.5889587, 452.4569397
1: -209.8780212, 176.3652649, -213.3681488, 179.2900391, -389.1680603, 389.7333374
2: -275.3728638, 177.9798737, -279.9508972, 180.9277039, -456.3005676, 457.9307556
3: -293.3232117, 154.6322174, -298.1987915, 157.1773987, -450.5006104, 452.8309631
4: -268.9162598, 205.0838013, -273.4079895, 208.4995575, -477.4158325, 478.4917908
5: -240.4125671, 186.3449554, -244.4365845, 189.4364014, -429.8489685, 430.7815552
6: -230.1435394, 221.5507507, -233.9858398, 225.2335358, -455.3770752, 455.5365906
7: -250.7793427, 209.9461670, -254.9468689, 213.4310913, -464.2104187, 464.8930359
8: -301.4733582, 205.3209381, -306.4979553, 208.7694855, -510.2428589, 511.8189087
9: -227.9011993, 223.8585358, -231.6973724, 227.5678711, -455.4690552, 455.5559082

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4074916, upper bound: 460.4075449
time: 9.98 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074379, upper bound: 460.4074382
time: 9.78 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -254.1189270, 201.6398621, -249.9490967, 198.3380127, -452.4569397, 451.5889587
1: -213.3681488, 179.2900391, -209.8780212, 176.3652649, -389.7333374, 389.1680603
2: -279.9508972, 180.9277039, -275.3728638, 177.9798737, -457.9307556, 456.3005676
3: -298.1987915, 157.1773987, -293.3232117, 154.6322174, -452.8309631, 450.5006104
4: -273.4079895, 208.4995575, -268.9162598, 205.0838013, -478.4917908, 477.4158325
5: -244.4365845, 189.4364014, -240.4125671, 186.3449554, -430.7815552, 429.8489685
6: -233.9858398, 225.2335358, -230.1435394, 221.5507507, -455.5365906, 455.3770752
7: -254.9468689, 213.4310913, -250.7793427, 209.9461670, -464.8930359, 464.2104187
8: -306.4979553, 208.7694855, -301.4733582, 205.3209381, -511.8189087, 510.2428589
9: -231.6973724, 227.5678711, -227.9011993, 223.8585358, -455.5559082, 455.4690552

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4074843, upper bound: 460.4075355
time: 10.85 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074362, upper bound: 460.4074362
time: 10.46 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -254.1189270, 201.6398621, -254.1189270, 201.6398621, -455.7587891, 455.7587891
1: -213.3681488, 179.2900391, -213.3681488, 179.2900391, -392.6582031, 392.6582031
2: -279.9508972, 180.9277039, -279.9508972, 180.9277039, -460.8786011, 460.8786011
3: -298.1987915, 157.1773987, -298.1987915, 157.1773987, -455.3761902, 455.3761902
4: -273.4079895, 208.4995575, -273.4079895, 208.4995575, -481.9075317, 481.9075317
5: -244.4365845, 189.4364014, -244.4365845, 189.4364014, -433.8729858, 433.8729858
6: -233.9858398, 225.2335358, -233.9858398, 225.2335358, -459.2193604, 459.2193604
7: -254.9468689, 213.4310913, -254.9468689, 213.4310913, -468.3779602, 468.3779602
8: -306.4979553, 208.7694855, -306.4979553, 208.7694855, -515.2674561, 515.2674561
9: -231.6973724, 227.5678711, -231.6973724, 227.5678711, -459.2652588, 459.2652588

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4074843, upper bound: 460.4075355
time: 10.21 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074361, upper bound: 460.4074361
time: 10.26 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 21.75 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.75
Output dim: 9, lower bound: -460.4074916, upper bound: 460.4075449
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 21.75
Output dim: 9, lower bound: -460.4074379, upper bound: 460.4074380
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.75
Output dim: 9, lower bound: -460.4074916, upper bound: 460.4075449
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 21.75
Output dim: 9, lower bound: -460.4074379, upper bound: 460.4074382
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.75
Output dim: 9, lower bound: -460.4074843, upper bound: 460.4075355
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 21.75
Output dim: 9, lower bound: -460.4074362, upper bound: 460.4074362
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.75
Output dim: 9, lower bound: -460.4074843, upper bound: 460.4075355
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 21.75
Output dim: 9, lower bound: -460.4074361, upper bound: 460.4074361

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -247.7831879, 196.6237335, -249.9490967, 198.3380127, -446.1212158, 446.5727844
1: -208.0666046, 174.8485260, -209.8780212, 176.3652649, -384.4318237, 384.7265625
2: -272.9868164, 176.4414978, -275.3728638, 177.9798737, -450.9666748, 451.8143616
3: -290.7887878, 153.3094330, -293.3232117, 154.6322174, -445.4209290, 446.6326294
4: -266.5844116, 203.3034973, -268.9162598, 205.0838013, -471.6682129, 472.2197571
5: -238.3409882, 184.7333832, -240.4125671, 186.3449554, -424.6858826, 425.1459351
6: -228.1599884, 219.6329803, -230.1435394, 221.5507507, -449.7107239, 449.7765198
7: -248.6087036, 208.1288147, -250.7793427, 209.9461670, -458.5548706, 458.9081421
8: -298.8659058, 203.5403442, -301.4733582, 205.3209381, -504.1868286, 505.0136414
9: -225.9190826, 221.9151459, -227.9011993, 223.8585358, -449.7776184, 449.8163452

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074410, upper bound: 460.4074410
time: 9.77 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074410, upper bound: 460.4074410
time: 10.17 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -247.7831879, 196.6237335, -254.1189270, 201.6398621, -449.4230347, 450.7426453
1: -208.0666046, 174.8485260, -213.3681488, 179.2900391, -387.3566284, 388.2166748
2: -272.9868164, 176.4414978, -279.9508972, 180.9277039, -453.9145203, 456.3923950
3: -290.7887878, 153.3094330, -298.1987915, 157.1773987, -447.9661560, 451.5082397
4: -266.5844116, 203.3034973, -273.4079895, 208.4995575, -475.0839844, 476.7114868
5: -238.3409882, 184.7333832, -244.4365845, 189.4364014, -427.7773743, 429.1699829
6: -228.1599884, 219.6329803, -233.9858398, 225.2335358, -453.3935242, 453.6188354
7: -248.6087036, 208.1288147, -254.9468689, 213.4310913, -462.0397949, 463.0756836
8: -298.8659058, 203.5403442, -306.4979553, 208.7694855, -507.6353760, 510.0382385
9: -225.9190826, 221.9151459, -231.6973724, 227.5678711, -453.4869385, 453.6125183

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074379, upper bound: 460.4074382
time: 10.07 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074379, upper bound: 460.4074382
time: 10.89 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -251.9547577, 199.9270782, -249.9490967, 198.3380127, -450.2927856, 449.8761597
1: -211.5583649, 177.7747040, -209.8780212, 176.3652649, -387.9236145, 387.6527100
2: -277.5670166, 179.3906555, -275.3728638, 177.9798737, -455.5468750, 454.7635193
3: -295.6665039, 155.8557281, -293.3232117, 154.6322174, -450.2987061, 449.1789551
4: -271.0782776, 206.7208862, -268.9162598, 205.0838013, -476.1620789, 475.6371460
5: -242.3665619, 187.8260651, -240.4125671, 186.3449554, -428.7114868, 428.2386475
6: -232.0039215, 223.3173828, -230.1435394, 221.5507507, -453.5546265, 453.4609375
7: -252.7781677, 211.6152496, -250.7793427, 209.9461670, -462.7243347, 462.3945923
8: -303.8927612, 206.9906006, -301.4733582, 205.3209381, -509.2136841, 508.4639587
9: -229.7169495, 225.6260376, -227.9011993, 223.8585358, -453.5754089, 453.5271912

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074382, upper bound: 460.4074378
time: 10.33 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074382, upper bound: 460.4074378
time: 9.60 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -251.9547577, 199.9270782, -254.1189270, 201.6398621, -453.5946045, 454.0460205
1: -211.5583649, 177.7747040, -213.3681488, 179.2900391, -390.8483887, 391.1428223
2: -277.5670166, 179.3906555, -279.9508972, 180.9277039, -458.4947205, 459.3415527
3: -295.6665039, 155.8557281, -298.1987915, 157.1773987, -452.8439026, 454.0545044
4: -271.0782776, 206.7208862, -273.4079895, 208.4995575, -479.5778198, 480.1288757
5: -242.3665619, 187.8260651, -244.4365845, 189.4364014, -431.8029480, 432.2626343
6: -232.0039215, 223.3173828, -233.9858398, 225.2335358, -457.2374573, 457.3032227
7: -252.7781677, 211.6152496, -254.9468689, 213.4310913, -466.2092285, 466.5621338
8: -303.8927612, 206.9906006, -306.4979553, 208.7694855, -512.6621704, 513.4885254
9: -229.7169495, 225.6260376, -231.6973724, 227.5678711, -457.2847595, 457.3234253

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074362, upper bound: 460.4074361
time: 8.56 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074362, upper bound: 460.4074361
time: 10.35 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 20.19 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 20.19
Output dim: 9, lower bound: -460.4074410, upper bound: 460.4074410
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 20.19
Output dim: 9, lower bound: -460.4074410, upper bound: 460.4074410
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 20.19
Output dim: 9, lower bound: -460.4074379, upper bound: 460.4074382
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 20.19
Output dim: 9, lower bound: -460.4074379, upper bound: 460.4074382
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 20.19
Output dim: 9, lower bound: -460.4074382, upper bound: 460.4074378
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 20.19
Output dim: 9, lower bound: -460.4074382, upper bound: 460.4074378
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 20.19
Output dim: 9, lower bound: -460.4074362, upper bound: 460.4074361
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 20.19
Output dim: 9, lower bound: -460.4074362, upper bound: 460.4074361
Binary search (step 1): status=Status.VERIFIED, k_low=8, k_high=12, k_mid=10, eps_mid=0.0390625, abs_max=463.9727783203125
rel_dist={9: [-460.4076338009184, 460.40763368826947]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076269, upper bound: 460.4076308
time: 9.56 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076210, upper bound: 460.4076209
time: 9.96 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 19.65 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 19.65
Output dim: 9, lower bound: -460.4076269, upper bound: 460.4076308
IS_A2, status: Status.UNKNOWN, split count: 1, time: 19.65
Output dim: 9, lower bound: -460.4076210, upper bound: 460.4076209

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -249.9490967, 198.3380127, -256.7159119, 203.7015533, -453.6506348, 455.0539246
1: -209.8780212, 176.3652649, -215.5436707, 181.1157990, -390.9938354, 391.9088745
2: -275.3728638, 177.9798737, -282.8133240, 182.7726288, -458.1454773, 460.7931824
3: -293.3232117, 154.6322174, -301.2575073, 158.7781830, -452.1013794, 455.8896790
4: -268.9162598, 205.0838013, -276.2066956, 210.6250763, -479.5413208, 481.2904968
5: -240.4125671, 186.3449554, -246.9300537, 191.3716736, -431.7842407, 433.2749634
6: -230.1435394, 221.5507507, -236.3738251, 227.5385132, -457.6820679, 457.9245605
7: -250.7793427, 209.9461670, -257.5447693, 215.6144562, -466.3937988, 467.4909058
8: -301.4733582, 205.3209381, -309.6375427, 210.9121857, -512.3855591, 514.9584961
9: -227.9011993, 223.8585358, -234.0735016, 229.8993225, -457.8004456, 457.9320374

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076210, upper bound: 460.4076209
time: 11.14 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076210, upper bound: 460.4076209
time: 9.62 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -254.1189270, 201.6398621, -256.7159119, 203.7015533, -457.8204956, 458.3557739
1: -213.3681488, 179.2900391, -215.5436707, 181.1157990, -394.4839478, 394.8337097
2: -279.9508972, 180.9277039, -282.8133240, 182.7726288, -462.7235107, 463.7410278
3: -298.1987915, 157.1773987, -301.2575073, 158.7781830, -456.9769897, 458.4349060
4: -273.4079895, 208.4995575, -276.2066956, 210.6250763, -484.0330200, 484.7062378
5: -244.4365845, 189.4364014, -246.9300537, 191.3716736, -435.8082581, 436.3664246
6: -233.9858398, 225.2335358, -236.3738251, 227.5385132, -461.5243530, 461.6073608
7: -254.9468689, 213.4310913, -257.5447693, 215.6144562, -470.5613403, 470.9757996
8: -306.4979553, 208.7694855, -309.6375427, 210.9121857, -517.4101562, 518.4070435
9: -231.6973724, 227.5678711, -234.0735016, 229.8993225, -461.5966492, 461.6413574

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076210, upper bound: 460.4076209
time: 10.71 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076210, upper bound: 460.4076210
time: 10.46 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.44 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 22.44
Output dim: 9, lower bound: -460.4076210, upper bound: 460.4076209
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 22.44
Output dim: 9, lower bound: -460.4076210, upper bound: 460.4076209
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 22.44
Output dim: 9, lower bound: -460.4076210, upper bound: 460.4076209
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 22.44
Output dim: 9, lower bound: -460.4076210, upper bound: 460.4076210

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -249.9490967, 198.3380127, -249.9490967, 198.3380127, -448.2871094, 448.2871094
1: -209.8780212, 176.3652649, -209.8780212, 176.3652649, -386.2432556, 386.2432556
2: -275.3728638, 177.9798737, -275.3728638, 177.9798737, -453.3527222, 453.3527222
3: -293.3232117, 154.6322174, -293.3232117, 154.6322174, -447.9553833, 447.9553833
4: -268.9162598, 205.0838013, -268.9162598, 205.0838013, -474.0000610, 474.0000610
5: -240.4125671, 186.3449554, -240.4125671, 186.3449554, -426.7575073, 426.7575073
6: -230.1435394, 221.5507507, -230.1435394, 221.5507507, -451.6942749, 451.6942749
7: -250.7793427, 209.9461670, -250.7793427, 209.9461670, -460.7255249, 460.7255249
8: -301.4733582, 205.3209381, -301.4733582, 205.3209381, -506.7943115, 506.7943115
9: -227.9011993, 223.8585358, -227.9011993, 223.8585358, -451.7597351, 451.7597351

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4074988, upper bound: 460.4075553
time: 10.31 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074418, upper bound: 460.4074421
time: 9.16 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -249.9490967, 198.3380127, -254.1189270, 201.6398621, -451.5889587, 452.4569397
1: -209.8780212, 176.3652649, -213.3681488, 179.2900391, -389.1680603, 389.7333374
2: -275.3728638, 177.9798737, -279.9508972, 180.9277039, -456.3005676, 457.9307556
3: -293.3232117, 154.6322174, -298.1987915, 157.1773987, -450.5006104, 452.8309631
4: -268.9162598, 205.0838013, -273.4079895, 208.4995575, -477.4158325, 478.4917908
5: -240.4125671, 186.3449554, -244.4365845, 189.4364014, -429.8489685, 430.7815552
6: -230.1435394, 221.5507507, -233.9858398, 225.2335358, -455.3770752, 455.5365906
7: -250.7793427, 209.9461670, -254.9468689, 213.4310913, -464.2104187, 464.8930359
8: -301.4733582, 205.3209381, -306.4979553, 208.7694855, -510.2428589, 511.8189087
9: -227.9011993, 223.8585358, -231.6973724, 227.5678711, -455.4690552, 455.5559082

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4074988, upper bound: 460.4075552
time: 10.52 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074418, upper bound: 460.4074420
time: 10.35 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -254.1189270, 201.6398621, -249.9490967, 198.3380127, -452.4569397, 451.5889587
1: -213.3681488, 179.2900391, -209.8780212, 176.3652649, -389.7333374, 389.1680603
2: -279.9508972, 180.9277039, -275.3728638, 177.9798737, -457.9307556, 456.3005676
3: -298.1987915, 157.1773987, -293.3232117, 154.6322174, -452.8309631, 450.5006104
4: -273.4079895, 208.4995575, -268.9162598, 205.0838013, -478.4917908, 477.4158325
5: -244.4365845, 189.4364014, -240.4125671, 186.3449554, -430.7815552, 429.8489685
6: -233.9858398, 225.2335358, -230.1435394, 221.5507507, -455.5365906, 455.3770752
7: -254.9468689, 213.4310913, -250.7793427, 209.9461670, -464.8930359, 464.2104187
8: -306.4979553, 208.7694855, -301.4733582, 205.3209381, -511.8189087, 510.2428589
9: -231.6973724, 227.5678711, -227.9011993, 223.8585358, -455.5559082, 455.4690552

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4074913, upper bound: 460.4075455
time: 10.42 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074400, upper bound: 460.4074400
time: 9.22 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -254.1189270, 201.6398621, -254.1189270, 201.6398621, -455.7587891, 455.7587891
1: -213.3681488, 179.2900391, -213.3681488, 179.2900391, -392.6582031, 392.6582031
2: -279.9508972, 180.9277039, -279.9508972, 180.9277039, -460.8786011, 460.8786011
3: -298.1987915, 157.1773987, -298.1987915, 157.1773987, -455.3761902, 455.3761902
4: -273.4079895, 208.4995575, -273.4079895, 208.4995575, -481.9075317, 481.9075317
5: -244.4365845, 189.4364014, -244.4365845, 189.4364014, -433.8729858, 433.8729858
6: -233.9858398, 225.2335358, -233.9858398, 225.2335358, -459.2193604, 459.2193604
7: -254.9468689, 213.4310913, -254.9468689, 213.4310913, -468.3779602, 468.3779602
8: -306.4979553, 208.7694855, -306.4979553, 208.7694855, -515.2674561, 515.2674561
9: -231.6973724, 227.5678711, -231.6973724, 227.5678711, -459.2652588, 459.2652588

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4074913, upper bound: 460.4075456
time: 9.65 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074400, upper bound: 460.4074400
time: 9.81 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 20.74 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 20.74
Output dim: 9, lower bound: -460.4074988, upper bound: 460.4075553
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 20.74
Output dim: 9, lower bound: -460.4074418, upper bound: 460.4074421
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 20.74
Output dim: 9, lower bound: -460.4074988, upper bound: 460.4075552
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 20.74
Output dim: 9, lower bound: -460.4074418, upper bound: 460.4074420
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 20.74
Output dim: 9, lower bound: -460.4074913, upper bound: 460.4075455
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 20.74
Output dim: 9, lower bound: -460.4074400, upper bound: 460.4074400
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 20.74
Output dim: 9, lower bound: -460.4074913, upper bound: 460.4075456
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 20.74
Output dim: 9, lower bound: -460.4074400, upper bound: 460.4074400

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -247.7831879, 196.6237335, -249.9490967, 198.3380127, -446.1212158, 446.5727844
1: -208.0666046, 174.8485260, -209.8780212, 176.3652649, -384.4318237, 384.7265625
2: -272.9868164, 176.4414978, -275.3728638, 177.9798737, -450.9666748, 451.8143616
3: -290.7887878, 153.3094330, -293.3232117, 154.6322174, -445.4209290, 446.6326294
4: -266.5844116, 203.3034973, -268.9162598, 205.0838013, -471.6682129, 472.2197571
5: -238.3409882, 184.7333832, -240.4125671, 186.3449554, -424.6858826, 425.1459351
6: -228.1599884, 219.6329803, -230.1435394, 221.5507507, -449.7107239, 449.7765198
7: -248.6087036, 208.1288147, -250.7793427, 209.9461670, -458.5548706, 458.9081421
8: -298.8659058, 203.5403442, -301.4733582, 205.3209381, -504.1868286, 505.0136414
9: -225.9190826, 221.9151459, -227.9011993, 223.8585358, -449.7776184, 449.8163452

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074448, upper bound: 460.4074448
time: 9.40 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074448, upper bound: 460.4074447
time: 9.21 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -247.7831879, 196.6237335, -254.1189270, 201.6398621, -449.4230347, 450.7426453
1: -208.0666046, 174.8485260, -213.3681488, 179.2900391, -387.3566284, 388.2166748
2: -272.9868164, 176.4414978, -279.9508972, 180.9277039, -453.9145203, 456.3923950
3: -290.7887878, 153.3094330, -298.1987915, 157.1773987, -447.9661560, 451.5082397
4: -266.5844116, 203.3034973, -273.4079895, 208.4995575, -475.0839844, 476.7114868
5: -238.3409882, 184.7333832, -244.4365845, 189.4364014, -427.7773743, 429.1699829
6: -228.1599884, 219.6329803, -233.9858398, 225.2335358, -453.3935242, 453.6188354
7: -248.6087036, 208.1288147, -254.9468689, 213.4310913, -462.0397949, 463.0756836
8: -298.8659058, 203.5403442, -306.4979553, 208.7694855, -507.6353760, 510.0382385
9: -225.9190826, 221.9151459, -231.6973724, 227.5678711, -453.4869385, 453.6125183

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074418, upper bound: 460.4074421
time: 9.67 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074418, upper bound: 460.4074421
time: 10.09 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -251.9547577, 199.9270782, -249.9490967, 198.3380127, -450.2927856, 449.8761597
1: -211.5583649, 177.7747040, -209.8780212, 176.3652649, -387.9236145, 387.6527100
2: -277.5670166, 179.3906555, -275.3728638, 177.9798737, -455.5468750, 454.7635193
3: -295.6665039, 155.8557281, -293.3232117, 154.6322174, -450.2987061, 449.1789551
4: -271.0782776, 206.7208862, -268.9162598, 205.0838013, -476.1620789, 475.6371460
5: -242.3665619, 187.8260651, -240.4125671, 186.3449554, -428.7114868, 428.2386475
6: -232.0039215, 223.3173828, -230.1435394, 221.5507507, -453.5546265, 453.4609375
7: -252.7781677, 211.6152496, -250.7793427, 209.9461670, -462.7243347, 462.3945923
8: -303.8927612, 206.9906006, -301.4733582, 205.3209381, -509.2136841, 508.4639587
9: -229.7169495, 225.6260376, -227.9011993, 223.8585358, -453.5754089, 453.5271912

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074421, upper bound: 460.4074418
time: 9.26 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074421, upper bound: 460.4074418
time: 9.36 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -251.9547577, 199.9270782, -254.1189270, 201.6398621, -453.5946045, 454.0460205
1: -211.5583649, 177.7747040, -213.3681488, 179.2900391, -390.8483887, 391.1428223
2: -277.5670166, 179.3906555, -279.9508972, 180.9277039, -458.4947205, 459.3415527
3: -295.6665039, 155.8557281, -298.1987915, 157.1773987, -452.8439026, 454.0545044
4: -271.0782776, 206.7208862, -273.4079895, 208.4995575, -479.5778198, 480.1288757
5: -242.3665619, 187.8260651, -244.4365845, 189.4364014, -431.8029480, 432.2626343
6: -232.0039215, 223.3173828, -233.9858398, 225.2335358, -457.2374573, 457.3032227
7: -252.7781677, 211.6152496, -254.9468689, 213.4310913, -466.2092285, 466.5621338
8: -303.8927612, 206.9906006, -306.4979553, 208.7694855, -512.6621704, 513.4885254
9: -229.7169495, 225.6260376, -231.6973724, 227.5678711, -457.2847595, 457.3234253

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074399, upper bound: 460.4074399
time: 10.25 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074399, upper bound: 460.4074400
time: 10.15 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 21.68 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 21.68
Output dim: 9, lower bound: -460.4074448, upper bound: 460.4074448
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 21.68
Output dim: 9, lower bound: -460.4074448, upper bound: 460.4074447
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 21.68
Output dim: 9, lower bound: -460.4074418, upper bound: 460.4074421
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 21.68
Output dim: 9, lower bound: -460.4074418, upper bound: 460.4074421
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 21.68
Output dim: 9, lower bound: -460.4074421, upper bound: 460.4074418
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 21.68
Output dim: 9, lower bound: -460.4074421, upper bound: 460.4074418
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 21.68
Output dim: 9, lower bound: -460.4074399, upper bound: 460.4074399
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 21.68
Output dim: 9, lower bound: -460.4074399, upper bound: 460.4074400
Binary search (step 2): status=Status.VERIFIED, k_low=11, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=463.9727783203125
rel_dist={9: [-460.40763907635636, 460.4076391076393]}

## Binary search (step 3) starts
Candidate k: 12, corresponding eps: 0.0468750


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076323, upper bound: 460.4076362
time: 8.85 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076260, upper bound: 460.4076259
time: 7.79 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 16.81 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 16.81
Output dim: 9, lower bound: -460.4076323, upper bound: 460.4076362
IS_A2, status: Status.UNKNOWN, split count: 1, time: 16.81
Output dim: 9, lower bound: -460.4076260, upper bound: 460.4076259

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -249.9490967, 198.3380127, -256.7159119, 203.7015533, -453.6506348, 455.0539246
1: -209.8780212, 176.3652649, -215.5436707, 181.1157990, -390.9938354, 391.9088745
2: -275.3728638, 177.9798737, -282.8133240, 182.7726288, -458.1454773, 460.7931824
3: -293.3232117, 154.6322174, -301.2575073, 158.7781830, -452.1013794, 455.8896790
4: -268.9162598, 205.0838013, -276.2066956, 210.6250763, -479.5413208, 481.2904968
5: -240.4125671, 186.3449554, -246.9300537, 191.3716736, -431.7842407, 433.2749634
6: -230.1435394, 221.5507507, -236.3738251, 227.5385132, -457.6820679, 457.9245605
7: -250.7793427, 209.9461670, -257.5447693, 215.6144562, -466.3937988, 467.4909058
8: -301.4733582, 205.3209381, -309.6375427, 210.9121857, -512.3855591, 514.9584961
9: -227.9011993, 223.8585358, -234.0735016, 229.8993225, -457.8004456, 457.9320374

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076260, upper bound: 460.4076258
time: 8.94 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076260, upper bound: 460.4076259
time: 8.71 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -254.1189270, 201.6398621, -256.7159119, 203.7015533, -457.8204956, 458.3557739
1: -213.3681488, 179.2900391, -215.5436707, 181.1157990, -394.4839478, 394.8337097
2: -279.9508972, 180.9277039, -282.8133240, 182.7726288, -462.7235107, 463.7410278
3: -298.1987915, 157.1773987, -301.2575073, 158.7781830, -456.9769897, 458.4349060
4: -273.4079895, 208.4995575, -276.2066956, 210.6250763, -484.0330200, 484.7062378
5: -244.4365845, 189.4364014, -246.9300537, 191.3716736, -435.8082581, 436.3664246
6: -233.9858398, 225.2335358, -236.3738251, 227.5385132, -461.5243530, 461.6073608
7: -254.9468689, 213.4310913, -257.5447693, 215.6144562, -470.5613403, 470.9757996
8: -306.4979553, 208.7694855, -309.6375427, 210.9121857, -517.4101562, 518.4070435
9: -231.6973724, 227.5678711, -234.0735016, 229.8993225, -461.5966492, 461.6413574

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076260, upper bound: 460.4076260
time: 7.96 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076260, upper bound: 460.4076259
time: 7.80 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 17.05 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 17.05
Output dim: 9, lower bound: -460.4076260, upper bound: 460.4076258
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 17.05
Output dim: 9, lower bound: -460.4076260, upper bound: 460.4076259
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 17.05
Output dim: 9, lower bound: -460.4076260, upper bound: 460.4076260
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 17.05
Output dim: 9, lower bound: -460.4076260, upper bound: 460.4076259

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -249.9490967, 198.3380127, -249.9490967, 198.3380127, -448.2871094, 448.2871094
1: -209.8780212, 176.3652649, -209.8780212, 176.3652649, -386.2432556, 386.2432556
2: -275.3728638, 177.9798737, -275.3728638, 177.9798737, -453.3527222, 453.3527222
3: -293.3232117, 154.6322174, -293.3232117, 154.6322174, -447.9553833, 447.9553833
4: -268.9162598, 205.0838013, -268.9162598, 205.0838013, -474.0000610, 474.0000610
5: -240.4125671, 186.3449554, -240.4125671, 186.3449554, -426.7575073, 426.7575073
6: -230.1435394, 221.5507507, -230.1435394, 221.5507507, -451.6942749, 451.6942749
7: -250.7793427, 209.9461670, -250.7793427, 209.9461670, -460.7255249, 460.7255249
8: -301.4733582, 205.3209381, -301.4733582, 205.3209381, -506.7943115, 506.7943115
9: -227.9011993, 223.8585358, -227.9011993, 223.8585358, -451.7597351, 451.7597351

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4075059, upper bound: 460.4075651
time: 8.02 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074456, upper bound: 460.4074458
time: 9.79 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -249.9490967, 198.3380127, -254.1189270, 201.6398621, -451.5889587, 452.4569397
1: -209.8780212, 176.3652649, -213.3681488, 179.2900391, -389.1680603, 389.7333374
2: -275.3728638, 177.9798737, -279.9508972, 180.9277039, -456.3005676, 457.9307556
3: -293.3232117, 154.6322174, -298.1987915, 157.1773987, -450.5006104, 452.8309631
4: -268.9162598, 205.0838013, -273.4079895, 208.4995575, -477.4158325, 478.4917908
5: -240.4125671, 186.3449554, -244.4365845, 189.4364014, -429.8489685, 430.7815552
6: -230.1435394, 221.5507507, -233.9858398, 225.2335358, -455.3770752, 455.5365906
7: -250.7793427, 209.9461670, -254.9468689, 213.4310913, -464.2104187, 464.8930359
8: -301.4733582, 205.3209381, -306.4979553, 208.7694855, -510.2428589, 511.8189087
9: -227.9011993, 223.8585358, -231.6973724, 227.5678711, -455.4690552, 455.5559082

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4075059, upper bound: 460.4075652
time: 9.37 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074456, upper bound: 460.4074459
time: 9.00 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -254.1189270, 201.6398621, -249.9490967, 198.3380127, -452.4569397, 451.5889587
1: -213.3681488, 179.2900391, -209.8780212, 176.3652649, -389.7333374, 389.1680603
2: -279.9508972, 180.9277039, -275.3728638, 177.9798737, -457.9307556, 456.3005676
3: -298.1987915, 157.1773987, -293.3232117, 154.6322174, -452.8309631, 450.5006104
4: -273.4079895, 208.4995575, -268.9162598, 205.0838013, -478.4917908, 477.4158325
5: -244.4365845, 189.4364014, -240.4125671, 186.3449554, -430.7815552, 429.8489685
6: -233.9858398, 225.2335358, -230.1435394, 221.5507507, -455.5365906, 455.3770752
7: -254.9468689, 213.4310913, -250.7793427, 209.9461670, -464.8930359, 464.2104187
8: -306.4979553, 208.7694855, -301.4733582, 205.3209381, -511.8189087, 510.2428589
9: -231.6973724, 227.5678711, -227.9011993, 223.8585358, -455.5559082, 455.4690552

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4074980, upper bound: 460.4075547
time: 8.93 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074438, upper bound: 460.4074438
time: 8.30 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -254.1189270, 201.6398621, -254.1189270, 201.6398621, -455.7587891, 455.7587891
1: -213.3681488, 179.2900391, -213.3681488, 179.2900391, -392.6582031, 392.6582031
2: -279.9508972, 180.9277039, -279.9508972, 180.9277039, -460.8786011, 460.8786011
3: -298.1987915, 157.1773987, -298.1987915, 157.1773987, -455.3761902, 455.3761902
4: -273.4079895, 208.4995575, -273.4079895, 208.4995575, -481.9075317, 481.9075317
5: -244.4365845, 189.4364014, -244.4365845, 189.4364014, -433.8729858, 433.8729858
6: -233.9858398, 225.2335358, -233.9858398, 225.2335358, -459.2193604, 459.2193604
7: -254.9468689, 213.4310913, -254.9468689, 213.4310913, -468.3779602, 468.3779602
8: -306.4979553, 208.7694855, -306.4979553, 208.7694855, -515.2674561, 515.2674561
9: -231.6973724, 227.5678711, -231.6973724, 227.5678711, -459.2652588, 459.2652588

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4074980, upper bound: 460.4075547
time: 8.41 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074438, upper bound: 460.4074437
time: 7.98 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 17.68 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.68
Output dim: 9, lower bound: -460.4075059, upper bound: 460.4075651
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 17.68
Output dim: 9, lower bound: -460.4074456, upper bound: 460.4074458
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.68
Output dim: 9, lower bound: -460.4075059, upper bound: 460.4075652
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 17.68
Output dim: 9, lower bound: -460.4074456, upper bound: 460.4074459
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.68
Output dim: 9, lower bound: -460.4074980, upper bound: 460.4075547
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 17.68
Output dim: 9, lower bound: -460.4074438, upper bound: 460.4074438
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.68
Output dim: 9, lower bound: -460.4074980, upper bound: 460.4075547
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 17.68
Output dim: 9, lower bound: -460.4074438, upper bound: 460.4074437

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -247.7831879, 196.6237335, -249.9490967, 198.3380127, -446.1212158, 446.5727844
1: -208.0666046, 174.8485260, -209.8780212, 176.3652649, -384.4318237, 384.7265625
2: -272.9868164, 176.4414978, -275.3728638, 177.9798737, -450.9666748, 451.8143616
3: -290.7887878, 153.3094330, -293.3232117, 154.6322174, -445.4209290, 446.6326294
4: -266.5844116, 203.3034973, -268.9162598, 205.0838013, -471.6682129, 472.2197571
5: -238.3409882, 184.7333832, -240.4125671, 186.3449554, -424.6858826, 425.1459351
6: -228.1599884, 219.6329803, -230.1435394, 221.5507507, -449.7107239, 449.7765198
7: -248.6087036, 208.1288147, -250.7793427, 209.9461670, -458.5548706, 458.9081421
8: -298.8659058, 203.5403442, -301.4733582, 205.3209381, -504.1868286, 505.0136414
9: -225.9190826, 221.9151459, -227.9011993, 223.8585358, -449.7776184, 449.8163452

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074485, upper bound: 460.4074484
time: 7.95 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074485, upper bound: 460.4074484
time: 9.52 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -247.7831879, 196.6237335, -254.1189270, 201.6398621, -449.4230347, 450.7426453
1: -208.0666046, 174.8485260, -213.3681488, 179.2900391, -387.3566284, 388.2166748
2: -272.9868164, 176.4414978, -279.9508972, 180.9277039, -453.9145203, 456.3923950
3: -290.7887878, 153.3094330, -298.1987915, 157.1773987, -447.9661560, 451.5082397
4: -266.5844116, 203.3034973, -273.4079895, 208.4995575, -475.0839844, 476.7114868
5: -238.3409882, 184.7333832, -244.4365845, 189.4364014, -427.7773743, 429.1699829
6: -228.1599884, 219.6329803, -233.9858398, 225.2335358, -453.3935242, 453.6188354
7: -248.6087036, 208.1288147, -254.9468689, 213.4310913, -462.0397949, 463.0756836
8: -298.8659058, 203.5403442, -306.4979553, 208.7694855, -507.6353760, 510.0382385
9: -225.9190826, 221.9151459, -231.6973724, 227.5678711, -453.4869385, 453.6125183

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074456, upper bound: 460.4074459
time: 9.34 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074456, upper bound: 460.4074458
time: 8.44 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -251.9547577, 199.9270782, -249.9490967, 198.3380127, -450.2927856, 449.8761597
1: -211.5583649, 177.7747040, -209.8780212, 176.3652649, -387.9236145, 387.6527100
2: -277.5670166, 179.3906555, -275.3728638, 177.9798737, -455.5468750, 454.7635193
3: -295.6665039, 155.8557281, -293.3232117, 154.6322174, -450.2987061, 449.1789551
4: -271.0782776, 206.7208862, -268.9162598, 205.0838013, -476.1620789, 475.6371460
5: -242.3665619, 187.8260651, -240.4125671, 186.3449554, -428.7114868, 428.2386475
6: -232.0039215, 223.3173828, -230.1435394, 221.5507507, -453.5546265, 453.4609375
7: -252.7781677, 211.6152496, -250.7793427, 209.9461670, -462.7243347, 462.3945923
8: -303.8927612, 206.9906006, -301.4733582, 205.3209381, -509.2136841, 508.4639587
9: -229.7169495, 225.6260376, -227.9011993, 223.8585358, -453.5754089, 453.5271912

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074459, upper bound: 460.4074456
time: 8.29 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074459, upper bound: 460.4074456
time: 8.59 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -251.9547577, 199.9270782, -254.1189270, 201.6398621, -453.5946045, 454.0460205
1: -211.5583649, 177.7747040, -213.3681488, 179.2900391, -390.8483887, 391.1428223
2: -277.5670166, 179.3906555, -279.9508972, 180.9277039, -458.4947205, 459.3415527
3: -295.6665039, 155.8557281, -298.1987915, 157.1773987, -452.8439026, 454.0545044
4: -271.0782776, 206.7208862, -273.4079895, 208.4995575, -479.5778198, 480.1288757
5: -242.3665619, 187.8260651, -244.4365845, 189.4364014, -431.8029480, 432.2626343
6: -232.0039215, 223.3173828, -233.9858398, 225.2335358, -457.2374573, 457.3032227
7: -252.7781677, 211.6152496, -254.9468689, 213.4310913, -466.2092285, 466.5621338
8: -303.8927612, 206.9906006, -306.4979553, 208.7694855, -512.6621704, 513.4885254
9: -229.7169495, 225.6260376, -231.6973724, 227.5678711, -457.2847595, 457.3234253

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074438, upper bound: 460.4074437
time: 8.07 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074438, upper bound: 460.4074437
time: 9.15 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 18.52 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 18.52
Output dim: 9, lower bound: -460.4074485, upper bound: 460.4074484
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 18.52
Output dim: 9, lower bound: -460.4074485, upper bound: 460.4074484
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 18.52
Output dim: 9, lower bound: -460.4074456, upper bound: 460.4074459
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 18.52
Output dim: 9, lower bound: -460.4074456, upper bound: 460.4074458
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 18.52
Output dim: 9, lower bound: -460.4074459, upper bound: 460.4074456
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 18.52
Output dim: 9, lower bound: -460.4074459, upper bound: 460.4074456
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 18.52
Output dim: 9, lower bound: -460.4074438, upper bound: 460.4074437
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 18.52
Output dim: 9, lower bound: -460.4074438, upper bound: 460.4074437
Binary search (step 3): status=Status.VERIFIED, k_low=12, k_high=12, k_mid=12, eps_mid=0.0468750, abs_max=463.9727783203125
rel_dist={9: [-460.40764437973144, 460.40764437529225]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.046875
execution time: 911.32 seconds
