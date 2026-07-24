## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 9.007671511800002


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.6137094, 5.8590560, -7.6137094, 5.8590560, -13.4727650, 13.4727640)
1: (-6.1292791, 5.3075094, -6.1292791, 5.3075094, -11.4367886, 11.4367886)
2: (-7.4691033, 4.5115442, -7.4691033, 4.5115442, -11.9806461, 11.9806480)
3: (-9.0654783, 4.4164257, -9.0654783, 4.4164257, -13.4819021, 13.4819012)
4: (-8.2032833, 6.3942995, -8.2032833, 6.3942995, -14.5975828, 14.5975819)
5: (-6.5569053, 5.5308518, -6.5569053, 5.5308518, -12.0877571, 12.0877571)
6: (-6.6569500, 7.0587082, -6.6569500, 7.0587082, -13.7156572, 13.7156572)
7: (-8.3428059, 4.1933599, -8.3428059, 4.1933599, -12.5361652, 12.5361652)
8: (-8.1921329, 6.0312138, -8.1921329, 6.0312138, -14.2233467, 14.2233467)
9: (-6.3373275, 6.7819953, -6.3373275, 6.7819953, -13.1193199, 13.1193218)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.50 + 4.98 = 7.48 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -9.0166882, upper bound: 9.0166883

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0139124, upper bound: 9.0099656
time: 5.95 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0095889, upper bound: 9.0095889
time: 3.82 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 10.04 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 10.04
Output dim: 7, lower bound: -9.0139124, upper bound: 9.0099656
NS_A2, status: Status.UNKNOWN, split count: 1, time: 10.04
Output dim: 7, lower bound: -9.0095889, upper bound: 9.0095889

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -7.5955572, 5.8445797, -7.6137094, 5.8590560, -13.4546127, 13.4582891
1: -6.1136093, 5.2946382, -6.1292791, 5.3075094, -11.4211159, 11.4239178
2: -7.4492431, 4.5000267, -7.4691033, 4.5115442, -11.9607868, 11.9691296
3: -9.0429592, 4.4055662, -9.0654783, 4.4164257, -13.4593840, 13.4710407
4: -8.1823597, 6.3783607, -8.2032833, 6.3942995, -14.5766592, 14.5816441
5: -6.5400386, 5.5173426, -6.5569053, 5.5308518, -12.0708904, 12.0742474
6: -6.6403832, 7.0413785, -6.6569500, 7.0587082, -13.6990910, 13.6983271
7: -8.3214378, 4.1806955, -8.3428059, 4.1933599, -12.5147972, 12.5235004
8: -8.1715746, 6.0167332, -8.1921329, 6.0312138, -14.2027884, 14.2088652
9: -6.3204532, 6.7643962, -6.3373275, 6.7819953, -13.1024475, 13.1017227

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0128106, upper bound: 9.0085603
time: 5.27 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0136287, upper bound: 9.0097746
time: 4.04 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -11.4618921, 8.7861767, -7.5985565, 5.8469539, -17.3088455, 16.3847332
1: -9.2981081, 7.8428025, -6.1161723, 5.2967691, -14.5948753, 13.9589748
2: -11.2832632, 6.5575161, -7.4525409, 4.5020156, -15.7852783, 14.0100574
3: -13.7841749, 6.4289460, -9.0466480, 4.4073906, -18.1915627, 15.4755917
4: -12.3545294, 9.4555264, -8.1858864, 6.3809571, -18.7354832, 17.6414127
5: -9.9948931, 8.1548386, -6.5427670, 5.5195389, -15.5144320, 14.6976032
6: -9.9219246, 10.5157146, -6.6431484, 7.0443134, -16.9662361, 17.1588631
7: -12.3882370, 6.1962557, -8.3251963, 4.1827192, -16.5709553, 14.5214520
8: -12.2955990, 8.8917942, -8.1749477, 6.0191197, -18.3147182, 17.0667419
9: -9.4355602, 10.0889034, -6.3230643, 6.7672729, -16.2028313, 16.4119682

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 72

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0090326, upper bound: 9.0081956
time: 2.18 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0094060, upper bound: 9.0094060
time: 2.40 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 13.76 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 13.76
Output dim: 7, lower bound: -9.0128106, upper bound: 9.0085603
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 13.76
Output dim: 7, lower bound: -9.0136287, upper bound: 9.0097746
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 13.76
Output dim: 7, lower bound: -9.0090326, upper bound: 9.0081956
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 13.76
Output dim: 7, lower bound: -9.0094060, upper bound: 9.0094060

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -7.5923748, 5.8419423, -7.4363236, 5.7155976, -13.3079720, 13.2782660
1: -6.1107802, 5.2922702, -5.9711609, 5.1793494, -11.2901278, 11.2634315
2: -7.4455948, 4.4978485, -7.2749095, 4.3961573, -11.8417521, 11.7727585
3: -9.0388641, 4.4034162, -8.8425083, 4.2966771, -13.3355408, 13.2459240
4: -8.1786327, 6.3753219, -8.0005665, 6.2316737, -14.4103069, 14.3758888
5: -6.5369520, 5.5147843, -6.3874593, 5.3953571, -11.9323092, 11.9022436
6: -6.6373873, 7.0383248, -6.4942737, 6.8905997, -13.5279856, 13.5325985
7: -8.3177099, 4.1779284, -8.1398649, 4.0495753, -12.3672848, 12.3177929
8: -8.1678791, 6.0140491, -7.9943476, 5.8853970, -14.0532761, 14.0083961
9: -6.3169584, 6.7610679, -6.1536164, 6.6015611, -12.9185200, 12.9146824

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 72

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0128106, upper bound: 9.0085603
time: 4.19 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0128106, upper bound: 9.0085603
time: 5.42 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -7.5884371, 5.8387451, -7.6525245, 5.8821526, -13.4705877, 13.4912682
1: -6.1073198, 5.2894077, -6.1535773, 5.3257508, -11.4330711, 11.4429846
2: -7.4411883, 4.4952965, -7.4937177, 4.5179243, -11.9591122, 11.9890137
3: -9.0338669, 4.4007907, -9.1112919, 4.4179034, -13.4517708, 13.5120831
4: -8.1741524, 6.3717308, -8.2390251, 6.4093227, -14.5834742, 14.6107559
5: -6.5331135, 5.5116839, -6.5827065, 5.5467153, -12.0798244, 12.0943909
6: -6.6337457, 7.0346417, -6.6816645, 7.0888405, -13.7225857, 13.7163048
7: -8.3132515, 4.1746616, -8.3745699, 4.1748061, -12.4880581, 12.5492306
8: -8.1633711, 6.0108495, -8.2279015, 6.0499687, -14.2133398, 14.2387505
9: -6.3128705, 6.7570658, -6.3381491, 6.7948503, -13.1077204, 13.0952129

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 57

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0135744, upper bound: 9.0097662
time: 5.51 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0135744, upper bound: 9.0097746
time: 4.30 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -11.4590921, 8.7839947, -7.4219718, 5.7042665, -17.1633587, 16.2059669
1: -9.2956934, 7.8408446, -5.9588928, 5.1693621, -14.4650545, 13.7997379
2: -11.2803421, 6.5557642, -7.2598996, 4.3873205, -15.6676617, 13.8156643
3: -13.7806768, 6.4271078, -8.8247700, 4.2883329, -18.0690098, 15.2518778
4: -12.3513479, 9.4531269, -7.9841533, 6.2193427, -18.5706863, 17.4372807
5: -9.9922781, 8.1527605, -6.3743463, 5.3849320, -15.3772106, 14.5271072
6: -9.9194155, 10.5131283, -6.4814320, 6.8769960, -16.7964096, 16.9945602
7: -12.3851204, 6.1942167, -8.1233196, 4.0406828, -16.4258041, 14.3175344
8: -12.2925129, 8.8896236, -7.9785438, 5.8741412, -18.1666527, 16.8681679
9: -9.4329290, 10.0862093, -6.1408749, 6.5884418, -16.0213699, 16.2270851

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0083784, upper bound: 9.0075662
time: 4.31 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0084077, upper bound: 9.0075675
time: 5.96 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -11.4556694, 8.7813435, -7.6380620, 5.8706594, -17.3263245, 16.4194050
1: -9.2927818, 7.8384976, -6.1410298, 5.3156662, -14.6084480, 13.9795265
2: -11.2768126, 6.5537090, -7.4782314, 4.5089769, -15.7857895, 14.0319395
3: -13.7764349, 6.4249015, -9.0934429, 4.4092674, -18.1857033, 15.5183449
4: -12.3475323, 9.4502487, -8.2225494, 6.3967052, -18.7442379, 17.6727982
5: -9.9890680, 8.1502419, -6.5694218, 5.5362449, -15.5253096, 14.7196636
6: -9.9163971, 10.5100307, -6.6686773, 7.0750532, -16.9914474, 17.1787071
7: -12.3814268, 6.1917791, -8.3578024, 4.1651478, -16.5465736, 14.5495806
8: -12.2887793, 8.8870096, -8.2119179, 6.0383840, -18.3271637, 17.0989265
9: -9.4297972, 10.0829716, -6.3248234, 6.7809925, -16.2107887, 16.4077950

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 29

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of NS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0091226, upper bound: 9.0090207
time: 5.33 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0090142, upper bound: 9.0090142
time: 2.52 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 17.05 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.05
Output dim: 7, lower bound: -9.0128106, upper bound: 9.0085603
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 17.05
Output dim: 7, lower bound: -9.0128106, upper bound: 9.0085603
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.05
Output dim: 7, lower bound: -9.0135744, upper bound: 9.0097662
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 17.05
Output dim: 7, lower bound: -9.0135744, upper bound: 9.0097746
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.05
Output dim: 7, lower bound: -9.0083784, upper bound: 9.0075662
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 17.05
Output dim: 7, lower bound: -9.0084077, upper bound: 9.0075675
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.05
Output dim: 7, lower bound: -9.0091226, upper bound: 9.0090207
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 17.05
Output dim: 7, lower bound: -9.0090142, upper bound: 9.0090142

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -7.4193816, 5.7021632, -7.4363236, 5.7155976, -13.1349792, 13.1384869
1: -5.9566422, 5.1674819, -5.9711609, 5.1793494, -11.1359921, 11.1386433
2: -7.2570238, 4.3855672, -7.2749095, 4.3961573, -11.6531811, 11.6604767
3: -8.8215075, 4.2867560, -8.8425083, 4.2966771, -13.1181850, 13.1292648
4: -7.9811420, 6.2170606, -8.0005665, 6.2316737, -14.2128162, 14.2176266
5: -6.3719726, 5.3830128, -6.3874593, 5.3953571, -11.7673302, 11.7704716
6: -6.4790235, 6.8743858, -6.4942737, 6.8905997, -13.3696232, 13.3686600
7: -8.1199532, 4.0390301, -8.1398649, 4.0495753, -12.1695290, 12.1788950
8: -7.9756284, 5.8720083, -7.9943476, 5.8853970, -13.8610249, 13.8663559
9: -6.1385503, 6.5860109, -6.1536164, 6.6015611, -12.7401114, 12.7396278

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 72

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0123453, upper bound: 9.0079373
time: 3.79 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0123422, upper bound: 9.0079437
time: 3.27 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.6346922, 5.8679733, -7.4363236, 5.7155976, -13.3502903, 13.3042965
1: -6.1381230, 5.3132582, -5.9711609, 5.1793494, -11.3174696, 11.2844191
2: -7.4745131, 4.5067263, -7.2749095, 4.3961573, -11.8706703, 11.7816353
3: -9.0892801, 4.4071541, -8.8425083, 4.2966771, -13.3859577, 13.2496605
4: -8.2185402, 6.3937540, -8.0005665, 6.2316737, -14.4502125, 14.3943205
5: -6.5663590, 5.5337629, -6.3874593, 5.3953571, -11.9617157, 11.9212227
6: -6.6655555, 7.0717354, -6.4942737, 6.8905997, -13.5561552, 13.5660095
7: -8.3535728, 4.1628456, -8.1398649, 4.0495753, -12.4031487, 12.3027096
8: -8.2081099, 6.0356793, -7.9943476, 5.8853970, -14.0935068, 14.0300274
9: -6.3218312, 6.7777224, -6.1536164, 6.6015611, -12.9233923, 12.9313374

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 61

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0112944, upper bound: 9.0082603
time: 4.96 seconds

## Relational analysis of NS_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 54

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0123112, upper bound: 9.0079391
time: 3.62 seconds

## Relational analysis of NS_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0123422, upper bound: 9.0079437
time: 5.69 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -7.4193816, 5.7021632, -7.6525245, 5.8821526, -13.3015347, 13.3546848
1: -5.9566422, 5.1674819, -6.1535773, 5.3257508, -11.2823925, 11.3210573
2: -7.2570238, 4.3855672, -7.4937177, 4.5179243, -11.7749481, 11.8792839
3: -8.8215075, 4.2867560, -9.1112919, 4.4179034, -13.2394104, 13.3980484
4: -7.9811420, 6.2170606, -8.2390251, 6.4093227, -14.3904648, 14.4560833
5: -6.3719726, 5.3830128, -6.5827065, 5.5467153, -11.9186878, 11.9657192
6: -6.4790235, 6.8743858, -6.6816645, 7.0888405, -13.5678635, 13.5560493
7: -8.1199532, 4.0390301, -8.3745699, 4.1748061, -12.2947578, 12.4136000
8: -7.9756284, 5.8720083, -8.2279015, 6.0499687, -14.0255966, 14.0999079
9: -6.1385503, 6.5860109, -6.3381491, 6.7948503, -12.9333992, 12.9241600

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 61

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0112944, upper bound: 9.0094873
time: 3.89 seconds

## Relational analysis of NS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 54

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0123453, upper bound: 9.0091399
time: 8.96 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0123422, upper bound: 9.0091690
time: 6.20 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7.6346922, 5.8679733, -7.6525245, 5.8821526, -13.5168428, 13.5204964
1: -6.1381230, 5.3132582, -6.1535773, 5.3257508, -11.4638739, 11.4668350
2: -7.4745131, 4.5067263, -7.4937177, 4.5179243, -11.9924374, 12.0004444
3: -9.0892801, 4.4071541, -9.1112919, 4.4179034, -13.5071831, 13.5184450
4: -8.2185402, 6.3937540, -8.2390251, 6.4093227, -14.6278610, 14.6327763
5: -6.5663590, 5.5337629, -6.5827065, 5.5467153, -12.1130733, 12.1164694
6: -6.6655555, 7.0717354, -6.6816645, 7.0888405, -13.7543926, 13.7533979
7: -8.3535728, 4.1628456, -8.3745699, 4.1748061, -12.5283794, 12.5374146
8: -8.2081099, 6.0356793, -8.2279015, 6.0499687, -14.2580786, 14.2635794
9: -6.3218312, 6.7777224, -6.3381491, 6.7948503, -13.1166801, 13.1158695

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 61

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0112944, upper bound: 9.0094908
time: 2.30 seconds

## Relational analysis of NS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0123453, upper bound: 9.0091273
time: 4.58 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0123422, upper bound: 9.0091549
time: 6.13 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -11.0533085, 8.4728231, -7.4198213, 5.7025990, -16.7559071, 15.8926449
1: -8.9547663, 7.5661964, -5.9570746, 5.1678867, -14.1226530, 13.5232716
2: -10.8701487, 6.3273549, -7.2577324, 4.3860950, -15.2562437, 13.5850868
3: -13.2700558, 6.1797357, -8.8220329, 4.2870278, -17.5570831, 15.0017681
4: -11.9081030, 9.1207609, -7.9818001, 6.2175517, -18.1256542, 17.1025620
5: -9.6199284, 7.8643222, -6.3723769, 5.3833866, -15.0033150, 14.2366991
6: -9.5676718, 10.1487761, -6.4795518, 6.8750486, -16.4427204, 16.6283283
7: -11.9579077, 5.9389839, -8.1210318, 4.0393600, -15.9972677, 14.0600157
8: -11.8565874, 8.5830193, -7.9762430, 5.8724957, -17.7290821, 16.5592613
9: -9.0826244, 9.7187929, -6.1389980, 6.5865297, -15.6691542, 15.8577909

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0083784, upper bound: 9.0075562
time: 2.35 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0083784, upper bound: 9.0075657
time: 2.01 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -11.2464590, 8.6212597, -7.4006720, 5.6877961, -16.9342556, 16.0219307
1: -9.1164093, 7.6961255, -5.9408455, 5.1546602, -14.2710676, 13.6369705
2: -11.0653582, 6.4351859, -7.2383490, 4.3750620, -15.4404202, 13.6735344
3: -13.5093527, 6.2863898, -8.7975254, 4.2752795, -17.7846317, 15.0839138
4: -12.1183882, 9.2779636, -7.9607000, 6.2015009, -18.3198853, 17.2386627
5: -9.7950773, 7.9993744, -6.3549752, 5.3696089, -15.1646862, 14.3543491
6: -9.7342577, 10.3246508, -6.4627323, 6.8576722, -16.5919285, 16.7873840
7: -12.1667547, 6.0509129, -8.1006193, 4.0273948, -16.1941490, 14.1515322
8: -12.0645151, 8.7292757, -7.9556456, 5.8577995, -17.9223137, 16.6849213
9: -9.2456188, 9.8909664, -6.1220460, 6.5693445, -15.8149633, 16.0130119

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 54

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 61

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0084077, upper bound: 9.0075568
time: 3.30 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0084077, upper bound: 9.0075675
time: 2.31 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -10.8261642, 8.3009071, -7.6380620, 5.8706594, -16.6968193, 15.9389687
1: -8.7721119, 7.4214058, -6.1410298, 5.3156662, -14.0877762, 13.5624342
2: -10.6480465, 6.2106442, -7.4782314, 4.5089769, -15.1570215, 13.6888742
3: -13.0088406, 6.0932913, -9.0934429, 4.4092674, -17.4181080, 15.1867342
4: -11.6631784, 8.9431324, -8.2225494, 6.3967052, -18.0598831, 17.1656799
5: -9.4242382, 7.7166719, -6.5694218, 5.5362449, -14.9604826, 14.2860937
6: -9.3797398, 9.9421654, -6.6686773, 7.0750532, -16.4547920, 16.6108418
7: -11.7136784, 5.8486576, -8.3578024, 4.1651478, -15.8788233, 14.2064590
8: -11.6127033, 8.4161406, -8.2119179, 6.0383840, -17.6510868, 16.6280575
9: -8.9090204, 9.5334072, -6.3248234, 6.7809925, -15.6900129, 15.8582306

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 29

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 61

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 54

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 61

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0085054, upper bound: 9.0083812
time: 2.15 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0085093, upper bound: 9.0084109
time: 1.68 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -10.8493872, 8.3194418, -7.5665450, 5.8160172, -16.6654053, 15.8859844
1: -8.7922859, 7.4370656, -6.0817170, 5.2680206, -14.0603056, 13.5187817
2: -10.6730623, 6.2256188, -7.4066758, 4.4699020, -15.1429625, 13.6322937
3: -13.0345373, 6.1015472, -9.0055580, 4.3709354, -17.4054699, 15.1071033
4: -11.6900930, 8.9627151, -8.1448746, 6.3388147, -18.0289078, 17.1075897
5: -9.4454641, 7.7330799, -6.5054722, 5.4869394, -14.9324026, 14.2385521
6: -9.4000587, 9.9656868, -6.6074510, 7.0104451, -16.4105034, 16.5731373
7: -11.7429361, 5.8620586, -8.2817936, 4.1260409, -15.8689756, 14.1438522
8: -11.6399784, 8.4341755, -8.1352835, 5.9846072, -17.6245861, 16.5694561
9: -8.9291668, 9.5555773, -6.2653894, 6.7184820, -15.6476488, 15.8209658

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 61

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 54

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 61

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0075130, upper bound: 9.0084922
time: 1.48 seconds

## Relational analysis of NS_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0075130, upper bound: 9.0090140
time: 2.43 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 6.32 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.32
Output dim: 7, lower bound: -9.0123453, upper bound: 9.0079373
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.32
Output dim: 7, lower bound: -9.0123422, upper bound: 9.0079437
NS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 6.32
Output dim: 7, lower bound: -9.0123112, upper bound: 9.0079391
NS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 6.32
Output dim: 7, lower bound: -9.0123422, upper bound: 9.0079437
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.32
Output dim: 7, lower bound: -9.0123453, upper bound: 9.0091399
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.32
Output dim: 7, lower bound: -9.0123422, upper bound: 9.0091690
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.32
Output dim: 7, lower bound: -9.0123453, upper bound: 9.0091273
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.32
Output dim: 7, lower bound: -9.0123422, upper bound: 9.0091549
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.32
Output dim: 7, lower bound: -9.0083784, upper bound: 9.0075562
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.32
Output dim: 7, lower bound: -9.0083784, upper bound: 9.0075657
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.32
Output dim: 7, lower bound: -9.0084077, upper bound: 9.0075568
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.32
Output dim: 7, lower bound: -9.0084077, upper bound: 9.0075675
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.32
Output dim: 7, lower bound: -9.0085054, upper bound: 9.0083812
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.32
Output dim: 7, lower bound: -9.0085093, upper bound: 9.0084109
NS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 6.32
Output dim: 7, lower bound: -9.0075130, upper bound: 9.0084922
NS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 6.32
Output dim: 7, lower bound: -9.0075130, upper bound: 9.0090140

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -7.4172268, 5.7004919, -7.0473199, 5.4154477, -12.8326740, 12.7478123
1: -5.9548221, 5.1660037, -5.6427326, 4.9136367, -10.8684587, 10.8087368
2: -7.2548528, 4.3843393, -6.8826928, 4.1755285, -11.4303818, 11.2670326
3: -8.8187656, 4.2854500, -8.3512917, 4.0744028, -12.8931684, 12.6367416
4: -7.9787803, 6.2152658, -7.5719051, 5.9096155, -13.8883953, 13.7871704
5: -6.3699994, 5.3814654, -6.0357161, 5.1193886, -11.4893875, 11.4171810
6: -6.4771390, 6.8724351, -6.1560755, 6.5356712, -13.0128098, 13.0285110
7: -8.1176643, 4.0377059, -7.7226095, 3.8198938, -11.9375582, 11.7603149
8: -7.9733229, 5.8703599, -7.5738282, 5.5890718, -13.5623951, 13.4441881
9: -6.1366677, 6.5840964, -5.8172183, 6.2563024, -12.3929701, 12.4013147

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 57

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 61

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of NS_A1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0107860, upper bound: 9.0076289
time: 4.03 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0123550, upper bound: 9.0079386
time: 5.36 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0123550, upper bound: 9.0079436
time: 7.77 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -7.3980994, 5.6857057, -7.2471910, 5.5694709, -12.9675703, 12.9328966
1: -5.9386067, 5.1527886, -5.8097682, 5.0480947, -10.9867020, 10.9625568
2: -7.2354846, 4.3733172, -7.0840812, 4.2869644, -11.5224495, 11.4573984
3: -8.7942772, 4.2737122, -8.5984039, 4.1802235, -12.9745007, 12.8721161
4: -7.9577003, 6.1992307, -7.7904172, 6.0723829, -14.0300827, 13.9896479
5: -6.3526220, 5.3677030, -6.2154026, 5.2584143, -11.6110363, 11.5831051
6: -6.4603372, 6.8550735, -6.3280654, 6.7190976, -13.1794348, 13.1831388
7: -8.0972681, 4.0257578, -7.9400043, 3.9287219, -12.0259895, 11.9657621
8: -7.9527440, 5.8556752, -7.7898188, 5.7405052, -13.6932487, 13.6454945
9: -6.1197314, 6.5669222, -5.9841619, 6.4305439, -12.5502758, 12.5510845

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 57

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 61

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of NS_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0107592, upper bound: 9.0076292
time: 4.40 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 54

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0123550, upper bound: 9.0079464
time: 6.48 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0123550, upper bound: 9.0079527
time: 5.15 seconds

## BFS NS instance: NS_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -7.2389727, 5.5630102, -7.4341712, 5.7139301, -12.9529028, 12.9971809
1: -5.8036680, 5.0430980, -5.9693418, 5.1778736, -10.9815416, 11.0124397
2: -7.0746431, 4.2823505, -7.2727423, 4.3949304, -11.4695740, 11.5550928
3: -8.5903301, 4.1791224, -8.8397694, 4.2953706, -12.8857002, 13.0188923
4: -7.7823076, 6.0659208, -7.9982095, 6.2298813, -14.0121889, 14.0641308
5: -6.2085361, 5.2532220, -6.3854880, 5.3938122, -11.6023483, 11.6387100
6: -6.3211918, 6.7112050, -6.4923921, 6.8886490, -13.2098408, 13.2035971
7: -7.9295130, 3.9255381, -8.1375809, 4.0482512, -11.9777641, 12.0631189
8: -7.7804246, 5.7341223, -7.9920444, 5.8837495, -13.6641741, 13.7261667
9: -5.9784865, 6.4232306, -6.1517353, 6.5996475, -12.5781345, 12.5749664

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 29

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 61

### Candidate
type: A, layer: 1, pos: 61

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 54

### Candidate
type: B, layer: 1, pos: 54

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0123065, upper bound: 9.0079311
time: 4.49 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0123065, upper bound: 9.0079384
time: 7.85 seconds

## BFS NS instance: NS_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -7.4367933, 5.7155194, -7.4149437, 5.6990442, -13.1358376, 13.1304626
1: -5.9693127, 5.1764627, -5.9530492, 5.1646080, -11.1339207, 11.1295118
2: -7.2740884, 4.3929806, -7.2533021, 4.3838549, -11.6579437, 11.6462822
3: -8.8348827, 4.2843266, -8.8152037, 4.2835703, -13.1184530, 13.0995302
4: -7.9988441, 6.2275128, -7.9770656, 6.2137647, -14.2126083, 14.2045784
5: -6.3864961, 5.3912411, -6.3679352, 5.3799500, -11.7664461, 11.7591763
6: -6.4918327, 6.8927307, -6.4755211, 6.8712163, -13.3630486, 13.3682518
7: -8.1449518, 4.0345974, -8.1171093, 4.0361824, -12.1811342, 12.1517067
8: -7.9941211, 5.8842411, -7.9713988, 5.8689981, -13.8631191, 13.8556404
9: -6.1449518, 6.5966330, -6.1347108, 6.5824108, -12.7273626, 12.7313442

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 61

### Candidate
type: A, layer: 1, pos: 61

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 54

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0123422, upper bound: 9.0079351
time: 5.85 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0123422, upper bound: 9.0079437
time: 3.65 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -7.4172268, 5.7004919, -7.2557211, 5.5762873, -12.9935141, 12.9562130
1: -5.9548221, 5.1660037, -5.8180184, 5.0548301, -11.0096521, 10.9840221
2: -7.2548528, 4.3843393, -7.0923519, 4.2928352, -11.5476875, 11.4766912
3: -8.8187656, 4.2854500, -8.6110897, 4.1888218, -13.0075874, 12.8965397
4: -7.9787803, 6.2152658, -7.8016205, 6.0803370, -14.0591173, 14.0168858
5: -6.3699994, 5.3814654, -6.2237630, 5.2653756, -11.6353750, 11.6052284
6: -6.4771390, 6.8724351, -6.3362565, 6.7273030, -13.2044420, 13.2086916
7: -8.1176643, 4.0377059, -7.9492846, 3.9357767, -12.0534410, 11.9869900
8: -7.9733229, 5.8703599, -7.7990208, 5.7473426, -13.7206650, 13.6693802
9: -6.1366677, 6.5840964, -5.9932928, 6.4385624, -12.5752296, 12.5773888

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 29

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 61

### Candidate
type: B, layer: 1, pos: 54

### Candidate
type: A, layer: 1, pos: 61

### Candidate
type: A, layer: 1, pos: 54

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0131152, upper bound: 9.0091346
time: 15.43 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0131152, upper bound: 9.0091399
time: 4.51 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -7.3980994, 5.6857057, -7.4538527, 5.7290359, -13.1271353, 13.1395588
1: -5.9386067, 5.1527886, -5.9839449, 5.1884160, -11.1270227, 11.1367340
2: -7.2354846, 4.3733172, -7.2921095, 4.4036579, -11.6391430, 11.6654263
3: -8.7942772, 4.2737122, -8.8559771, 4.2942019, -13.0884790, 13.1296892
4: -7.9577003, 6.1992307, -8.0185032, 6.2422180, -14.1999187, 14.2177334
5: -6.3526220, 5.3677030, -6.4020081, 5.4036365, -11.7562580, 11.7697105
6: -6.4603372, 6.8550735, -6.5071898, 6.9091001, -13.3694372, 13.3622627
7: -8.0972681, 4.0257578, -8.1650715, 4.0451040, -12.1423721, 12.1908293
8: -7.9527440, 5.8556752, -8.0130463, 5.8977056, -13.8504496, 13.8687210
9: -6.1197314, 6.5669222, -6.1601191, 6.6123052, -12.7320366, 12.7270412

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 61

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 61

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 54

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0131151, upper bound: 9.0091616
time: 3.30 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0131151, upper bound: 9.0091690
time: 6.78 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -7.6325412, 5.8663025, -7.2557211, 5.5762873, -13.2088280, 13.1220236
1: -6.1362858, 5.3117762, -5.8180184, 5.0548301, -11.1911154, 11.1297951
2: -7.4723005, 4.5054913, -7.0923519, 4.2928352, -11.7651348, 11.5978432
3: -9.0865374, 4.4058285, -8.6110897, 4.1888218, -13.2753592, 13.0169182
4: -8.2161713, 6.3919401, -7.8016205, 6.0803370, -14.2965088, 14.1935606
5: -6.5644011, 5.5322242, -6.2237630, 5.2653756, -11.8297768, 11.7559872
6: -6.6636634, 7.0697699, -6.3362565, 6.7273030, -13.3909664, 13.4060268
7: -8.3512640, 4.1614704, -7.9492846, 3.9357767, -12.2870407, 12.1107550
8: -8.2057886, 6.0340075, -7.7990208, 5.7473426, -13.9531288, 13.8330288
9: -6.3199029, 6.7757425, -5.9932928, 6.4385624, -12.7584648, 12.7690334

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 57

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 61

### Candidate
type: A, layer: 1, pos: 61

### Candidate
type: B, layer: 1, pos: 54

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0130238, upper bound: 9.0091248
time: 4.80 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0130238, upper bound: 9.0091273
time: 6.42 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7.6128025, 5.8509703, -7.4538527, 5.7290359, -13.3418369, 13.3048229
1: -6.1193886, 5.2981153, -5.9839449, 5.1884160, -11.3078041, 11.2820606
2: -7.4519749, 4.4940534, -7.2921095, 4.4036579, -11.8556309, 11.7861633
3: -9.0613174, 4.3934851, -8.8559771, 4.2942019, -13.3555193, 13.2494621
4: -8.1943913, 6.3752041, -8.0185032, 6.2422180, -14.4366093, 14.3937073
5: -6.5464168, 5.5180464, -6.4020081, 5.4036365, -11.9500504, 11.9200544
6: -6.6462626, 7.0517573, -6.5071898, 6.9091001, -13.5553627, 13.5589466
7: -8.3300924, 4.1485577, -8.1650715, 4.0451040, -12.3751965, 12.3136292
8: -8.1844597, 6.0186458, -8.0130463, 5.8977056, -14.0821648, 14.0316925
9: -6.3020096, 6.7574496, -6.1601191, 6.6123052, -12.9143143, 12.9175682

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 61

### Candidate
type: B, layer: 1, pos: 61

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 54

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0130238, upper bound: 9.0091506
time: 4.07 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0130238, upper bound: 9.0091548
time: 6.54 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -11.0533085, 8.4728231, -7.0335379, 5.4045753, -16.4578838, 15.5063610
1: -8.9547663, 7.5661964, -5.6309676, 4.9040546, -13.8588200, 13.1971645
2: -10.8701487, 6.3273549, -6.8682556, 4.1670303, -15.0371790, 13.1956100
3: -13.2700558, 6.1797357, -8.3342705, 4.0664940, -17.3365498, 14.5140038
4: -11.9081030, 9.1207609, -7.5560608, 5.8978190, -17.8059216, 16.6768227
5: -9.6199284, 7.8643222, -6.0232038, 5.1094346, -14.7293625, 13.8875256
6: -9.5676718, 10.1487761, -6.1437597, 6.5225520, -16.0902233, 16.2925358
7: -11.9579077, 5.9389839, -7.7066517, 3.8115478, -15.7694550, 13.6456356
8: -11.8565874, 8.5830193, -7.5585709, 5.5782828, -17.4348698, 16.1415901
9: -9.0826244, 9.7187929, -5.8051000, 6.2437668, -15.3263912, 15.5238914

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 54

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0078563, upper bound: 9.0069072
time: 2.52 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0077967, upper bound: 9.0069001
time: 3.24 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -11.0533085, 8.4728231, -7.2332306, 5.5584612, -16.6117706, 15.7060537
1: -8.9547663, 7.5661964, -5.7978544, 5.0383945, -13.9931602, 13.3640509
2: -10.8701487, 6.3273549, -7.0694599, 4.2783723, -15.1485214, 13.3968143
3: -13.2700558, 6.1797357, -8.5811558, 4.1722240, -17.4422798, 14.7608910
4: -11.9081030, 9.1207609, -7.7743645, 6.0604343, -17.9685364, 16.8951263
5: -9.6199284, 7.8643222, -6.2027316, 5.2483358, -14.8682642, 14.0670538
6: -9.5676718, 10.1487761, -6.3155951, 6.7058158, -16.2734871, 16.4643707
7: -11.9579077, 5.9389839, -7.9238644, 3.9202800, -15.8781872, 13.8628483
8: -11.8565874, 8.5830193, -7.7743711, 5.7295837, -17.5861664, 16.3573914
9: -9.0826244, 9.7187929, -5.9719067, 6.4178510, -15.5004749, 15.6906986

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 54

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0078563, upper bound: 9.0069113
time: 3.67 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0077967, upper bound: 9.0069042
time: 2.25 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -11.2464590, 8.6212597, -7.0335379, 5.4045753, -16.6510353, 15.6547976
1: -9.1164093, 7.6961255, -5.6309676, 4.9040546, -14.0204601, 13.3270931
2: -11.0653582, 6.4351859, -6.8682556, 4.1670303, -15.2323885, 13.3034420
3: -13.5093527, 6.2863898, -8.3342705, 4.0664940, -17.5758476, 14.6206579
4: -12.1183882, 9.2779636, -7.5560608, 5.8978190, -18.0162048, 16.8340244
5: -9.7950773, 7.9993744, -6.0232038, 5.1094346, -14.9045115, 14.0225782
6: -9.7342577, 10.3246508, -6.1437597, 6.5225520, -16.2568092, 16.4684105
7: -12.1667547, 6.0509129, -7.7066517, 3.8115478, -15.9783020, 13.7575626
8: -12.0645151, 8.7292757, -7.5585709, 5.5782828, -17.6427975, 16.2878456
9: -9.2456188, 9.8909664, -5.8051000, 6.2437668, -15.4893856, 15.6960659

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 54

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0078939, upper bound: 9.0069061
time: 2.20 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0078424, upper bound: 9.0068996
time: 2.86 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -11.2464590, 8.6212597, -7.2332306, 5.5584612, -16.8049164, 15.8544903
1: -9.1164093, 7.6961255, -5.7978544, 5.0383945, -14.1548023, 13.4939804
2: -11.0653582, 6.4351859, -7.0694599, 4.2783723, -15.3437290, 13.5046463
3: -13.5093527, 6.2863898, -8.5811558, 4.1722240, -17.6815758, 14.8675432
4: -12.1183882, 9.2779636, -7.7743645, 6.0604343, -18.1788216, 17.0523281
5: -9.7950773, 7.9993744, -6.2027316, 5.2483358, -15.0434122, 14.2021065
6: -9.7342577, 10.3246508, -6.3155951, 6.7058158, -16.4400711, 16.6402454
7: -12.1667547, 6.0509129, -7.9238644, 3.9202800, -16.0870342, 13.9747763
8: -12.0645151, 8.7292757, -7.7743711, 5.7295837, -17.7940979, 16.5036469
9: -9.2456188, 9.8909664, -5.9719067, 6.4178510, -15.6634693, 15.8628712

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 54

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 61

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0078939, upper bound: 9.0069106
time: 2.04 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0078424, upper bound: 9.0069040
time: 3.91 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -10.8240604, 8.2992935, -7.2423239, 5.5657334, -16.3897934, 15.5416174
1: -8.7703409, 7.4199791, -5.8065920, 5.0455275, -13.8158665, 13.2265711
2: -10.6459150, 6.2094569, -7.0783348, 4.2846069, -14.9305210, 13.2877922
3: -13.0061932, 6.0920024, -8.5945187, 4.1811757, -17.1873684, 14.6865196
4: -11.6608744, 8.9414015, -7.7862849, 6.0689044, -17.7297783, 16.7276859
5: -9.4223061, 7.7151718, -6.2116065, 5.2557216, -14.6780272, 13.9267788
6: -9.3779116, 9.9402781, -6.3243074, 6.7145600, -16.0924683, 16.2645855
7: -11.7114592, 5.8473282, -7.9337769, 3.9276986, -15.6391582, 13.7811041
8: -11.6104412, 8.4145498, -7.7842126, 5.7368636, -17.3473053, 16.1987629
9: -8.9071980, 9.5314989, -5.9815507, 6.4263892, -15.3335876, 15.5130501

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 29

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 54

### Candidate
type: B, layer: 1, pos: 61

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0071279, upper bound: 9.0079800
time: 2.65 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -9.0071279, upper bound: 9.0068603
time: 2.07 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -10.8046913, 8.2844419, -7.4400916, 5.7182026, -16.5228939, 15.7245331
1: -8.7540388, 7.4068365, -5.9721866, 5.1788573, -13.9328957, 13.3790226
2: -10.6263008, 6.1984367, -7.2777281, 4.3952074, -15.0215082, 13.4761648
3: -12.9818630, 6.0799408, -8.8389950, 4.2863421, -17.2682056, 14.9189339
4: -11.6396189, 8.9254532, -8.0027580, 6.2304492, -17.8700657, 16.9282112
5: -9.4044828, 7.7013106, -6.3895144, 5.3937001, -14.7981815, 14.0908251
6: -9.3610716, 9.9229202, -6.4948983, 6.8960409, -16.2571125, 16.4178181
7: -11.6911221, 5.8347893, -8.1491699, 4.0367150, -15.7278366, 13.9839573
8: -11.5896015, 8.3999090, -7.9978571, 5.8869424, -17.4765434, 16.3977661
9: -8.8902540, 9.5138264, -6.1479635, 6.5997415, -15.4899960, 15.6617889

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 57

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 61

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 54

### Candidate
type: A, layer: 1, pos: 61

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0071278, upper bound: 9.0080050
time: 2.52 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0071278, upper bound: 9.0084109
time: 2.30 seconds

## BFS NS instance: NS_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -10.6852531, 8.1919842, -7.5665450, 5.8160172, -16.5012703, 15.7585258
1: -8.6507778, 7.3218613, -6.0817170, 5.2680206, -13.9187965, 13.4035778
2: -10.5031757, 6.1239104, -7.4066758, 4.4699020, -14.9730749, 13.5305862
3: -12.8284292, 5.9956918, -9.0055580, 4.3709354, -17.1993618, 15.0012474
4: -11.5070848, 8.8235931, -8.1448746, 6.3388147, -17.8458977, 16.9684677
5: -9.2928038, 7.6122613, -6.5054722, 5.4869394, -14.7797422, 14.1177311
6: -9.2537851, 9.8138781, -6.6074510, 7.0104451, -16.2642307, 16.4213295
7: -11.5600967, 5.7459908, -8.2817936, 4.1260409, -15.6861334, 14.0277843
8: -11.4599104, 8.3069839, -8.1352835, 5.9846072, -17.4445171, 16.4422646
9: -8.7753277, 9.3998337, -6.2653894, 6.7184820, -15.4938087, 15.6652222

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 54

### Candidate
type: B, layer: 1, pos: 61

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A2_B2_A2_A1_B1

### Relational analysis result of NS_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0068390, upper bound: 9.0077967
time: 2.08 seconds

## Relational analysis of NS_A2_B2_A2_A1_B2

### Relational analysis result of NS_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0068391, upper bound: 9.0078424
time: 1.77 seconds

## BFS NS instance: NS_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -10.9746914, 8.4139795, -7.5665450, 5.8160172, -16.7907085, 15.9805222
1: -8.8933506, 7.5168943, -6.0817170, 5.2680206, -14.1613684, 13.5986109
2: -10.7941923, 6.2845507, -7.4066758, 4.4699020, -15.2640915, 13.6912260
3: -13.1888418, 6.1531725, -9.0055580, 4.3709354, -17.5597744, 15.1587276
4: -11.8229876, 9.0580072, -8.1448746, 6.3388147, -18.1618004, 17.2028809
5: -9.5529442, 7.8134418, -6.5054722, 5.4869394, -15.0398817, 14.3189125
6: -9.5029831, 10.0787344, -6.6074510, 7.0104451, -16.5134277, 16.6861858
7: -11.8735676, 5.9065933, -8.2817936, 4.1260409, -15.9996023, 14.1883869
8: -11.7722769, 8.5258579, -8.1352835, 5.9846072, -17.7568836, 16.6611385
9: -9.0179510, 9.6554985, -6.2653894, 6.7184820, -15.7364302, 15.9208870

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 72

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 54

### Candidate
type: B, layer: 1, pos: 61

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A2_B2_A2_A2_B1

### Relational analysis result of NS_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0068390, upper bound: 9.0083738
time: 2.37 seconds

## Relational analysis of NS_A2_B2_A2_A2_B2

### Relational analysis result of NS_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0068391, upper bound: 9.0084044
time: 2.48 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 7.40 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.40
Output dim: 7, lower bound: -9.0123550, upper bound: 9.0079386
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.40
Output dim: 7, lower bound: -9.0123550, upper bound: 9.0079436
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.40
Output dim: 7, lower bound: -9.0123550, upper bound: 9.0079464
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.40
Output dim: 7, lower bound: -9.0123550, upper bound: 9.0079527
NS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 7.40
Output dim: 7, lower bound: -9.0123065, upper bound: 9.0079311
NS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 7.40
Output dim: 7, lower bound: -9.0123065, upper bound: 9.0079384
NS_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 7.40
Output dim: 7, lower bound: -9.0123422, upper bound: 9.0079351
NS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 7.40
Output dim: 7, lower bound: -9.0123422, upper bound: 9.0079437
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.40
Output dim: 7, lower bound: -9.0131152, upper bound: 9.0091346
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.40
Output dim: 7, lower bound: -9.0131152, upper bound: 9.0091399
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.40
Output dim: 7, lower bound: -9.0131151, upper bound: 9.0091616
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.40
Output dim: 7, lower bound: -9.0131151, upper bound: 9.0091690
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.40
Output dim: 7, lower bound: -9.0130238, upper bound: 9.0091248
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.40
Output dim: 7, lower bound: -9.0130238, upper bound: 9.0091273
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.40
Output dim: 7, lower bound: -9.0130238, upper bound: 9.0091506
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.40
Output dim: 7, lower bound: -9.0130238, upper bound: 9.0091548
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.40
Output dim: 7, lower bound: -9.0078563, upper bound: 9.0069072
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.40
Output dim: 7, lower bound: -9.0077967, upper bound: 9.0069001
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.40
Output dim: 7, lower bound: -9.0078563, upper bound: 9.0069113
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.40
Output dim: 7, lower bound: -9.0077967, upper bound: 9.0069042
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.40
Output dim: 7, lower bound: -9.0078939, upper bound: 9.0069061
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.40
Output dim: 7, lower bound: -9.0078424, upper bound: 9.0068996
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.40
Output dim: 7, lower bound: -9.0078939, upper bound: 9.0069106
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.40
Output dim: 7, lower bound: -9.0078424, upper bound: 9.0069040
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.40
Output dim: 7, lower bound: -9.0071279, upper bound: 9.0079800
NS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 7.40
Output dim: 7, lower bound: -9.0071279, upper bound: 9.0068603
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.40
Output dim: 7, lower bound: -9.0071278, upper bound: 9.0080050
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.40
Output dim: 7, lower bound: -9.0071278, upper bound: 9.0084109
NS_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 7.40
Output dim: 7, lower bound: -9.0068390, upper bound: 9.0077967
NS_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 7.40
Output dim: 7, lower bound: -9.0068391, upper bound: 9.0078424
NS_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 7.40
Output dim: 7, lower bound: -9.0068390, upper bound: 9.0083738
NS_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 7.40
Output dim: 7, lower bound: -9.0068391, upper bound: 9.0084044

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -7.0309415, 5.4024701, -7.0473199, 5.4154477, -12.4463892, 12.4497900
1: -5.6287084, 4.9021740, -5.6427326, 4.9136367, -10.5423450, 10.5449066
2: -6.8653851, 4.1652822, -6.8826928, 4.1755285, -11.0409136, 11.0479755
3: -8.3310232, 4.0649376, -8.3512917, 4.0744028, -12.4054260, 12.4162292
4: -7.5530319, 5.8955359, -7.5719051, 5.9096155, -13.4626474, 13.4674416
5: -6.0208292, 5.1075163, -6.0357161, 5.1193886, -11.1402178, 11.1432323
6: -6.1413541, 6.5199361, -6.1560755, 6.5356712, -12.6770248, 12.6760120
7: -7.7032814, 3.8099103, -7.7226095, 3.8198938, -11.5231752, 11.5325203
8: -7.5556479, 5.5761518, -7.5738282, 5.5890718, -13.1447201, 13.1499805
9: -5.8027539, 6.2413211, -5.8172183, 6.2563024, -12.0590563, 12.0585394

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 61

### Candidate
type: A, layer: 1, pos: 61

### Candidate
type: B, layer: 1, pos: 54

### Candidate
type: A, layer: 1, pos: 54

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A1_B1_A1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0119164, upper bound: 9.0072896
time: 3.42 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0117409, upper bound: 9.0072609
time: 3.86 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.2305493, 5.5562963, -7.0473199, 5.4154477, -12.6459970, 12.6036167
1: -5.7955317, 5.0364585, -5.6427326, 4.9136367, -10.7091684, 10.6791916
2: -7.0665126, 4.2765837, -6.8826928, 4.1755285, -11.2420406, 11.1592770
3: -8.5778008, 4.1706429, -8.3512917, 4.0744028, -12.6522036, 12.5219345
4: -7.7712717, 6.0580959, -7.5719051, 5.9096155, -13.6808872, 13.6300011
5: -6.2002831, 5.2463655, -6.0357161, 5.1193886, -11.3196716, 11.2820816
6: -6.3131261, 6.7031245, -6.1560755, 6.5356712, -12.8487968, 12.8591995
7: -7.9203949, 3.9186063, -7.7226095, 3.8198938, -11.7402887, 11.6412163
8: -7.7713599, 5.7273903, -7.5738282, 5.5890718, -13.3604317, 13.3012180
9: -5.9694896, 6.4153366, -5.8172183, 6.2563024, -12.2257919, 12.2325554

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 61

### Candidate
type: A, layer: 1, pos: 61

### Candidate
type: B, layer: 1, pos: 54

### Candidate
type: A, layer: 1, pos: 54

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 113

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A1_B1_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0119164, upper bound: 9.0072954
time: 2.80 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0117409, upper bound: 9.0072692
time: 4.74 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -7.0309415, 5.4024701, -7.2471910, 5.5694709, -12.6004124, 12.6496611
1: -5.6287084, 4.9021740, -5.8097682, 5.0480947, -10.6768036, 10.7119427
2: -6.8653851, 4.1652822, -7.0840812, 4.2869644, -11.1523495, 11.2493629
3: -8.3310232, 4.0649376, -8.5984039, 4.1802235, -12.5112467, 12.6633415
4: -7.5530319, 5.8955359, -7.7904172, 6.0723829, -13.6254148, 13.6859531
5: -6.0208292, 5.1075163, -6.2154026, 5.2584143, -11.2792435, 11.3229189
6: -6.1413541, 6.5199361, -6.3280654, 6.7190976, -12.8604517, 12.8480015
7: -7.7032814, 3.8099103, -7.9400043, 3.9287219, -11.6320038, 11.7499142
8: -7.5556479, 5.5761518, -7.7898188, 5.7405052, -13.2961531, 13.3659706
9: -5.8027539, 6.2413211, -5.9841619, 6.4305439, -12.2332973, 12.2254829

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 61

### Candidate
type: A, layer: 1, pos: 61

### Candidate
type: B, layer: 1, pos: 54

### Candidate
type: A, layer: 1, pos: 54

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A1_B1_A1_B2_A1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0119062, upper bound: 9.0072935
time: 4.98 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0117286, upper bound: 9.0072639
time: 5.75 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7.2305493, 5.5562963, -7.2471910, 5.5694709, -12.8000202, 12.8034878
1: -5.7955317, 5.0364585, -5.8097682, 5.0480947, -10.8436260, 10.8462267
2: -7.0665126, 4.2765837, -7.0840812, 4.2869644, -11.3534775, 11.3606644
3: -8.5778008, 4.1706429, -8.5984039, 4.1802235, -12.7580242, 12.7690468
4: -7.7712717, 6.0580959, -7.7904172, 6.0723829, -13.8436546, 13.8485126
5: -6.2002831, 5.2463655, -6.2154026, 5.2584143, -11.4586973, 11.4617682
6: -6.3131261, 6.7031245, -6.3280654, 6.7190976, -13.0322237, 13.0311899
7: -7.9203949, 3.9186063, -7.9400043, 3.9287219, -11.8491173, 11.8586102
8: -7.7713599, 5.7273903, -7.7898188, 5.7405052, -13.5118656, 13.5172091
9: -5.9694896, 6.4153366, -5.9841619, 6.4305439, -12.4000340, 12.3994980

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 61

### Candidate
type: A, layer: 1, pos: 61

### Candidate
type: B, layer: 1, pos: 54

### Candidate
type: A, layer: 1, pos: 54

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of NS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 113

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A1_B1_A1_B2_A2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0119062, upper bound: 9.0072994
time: 7.08 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0117286, upper bound: 9.0072726
time: 6.48 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -7.2389727, 5.5630102, -7.0473199, 5.4154477, -12.6544209, 12.6103306
1: -5.8036680, 5.0430980, -5.6427326, 4.9136367, -10.7173042, 10.6858311
2: -7.0746431, 4.2823505, -6.8826928, 4.1755285, -11.2501717, 11.1650429
3: -8.5903301, 4.1791224, -8.3512917, 4.0744028, -12.6647329, 12.5304146
4: -7.7823076, 6.0659208, -7.5719051, 5.9096155, -13.6919231, 13.6378260
5: -6.2085361, 5.2532220, -6.0357161, 5.1193886, -11.3279247, 11.2889385
6: -6.3211918, 6.7112050, -6.1560755, 6.5356712, -12.8568630, 12.8672810
7: -7.9295130, 3.9255381, -7.7226095, 3.8198938, -11.7494068, 11.6481476
8: -7.7804246, 5.7341223, -7.5738282, 5.5890718, -13.3694963, 13.3079510
9: -5.9784865, 6.4232306, -5.8172183, 6.2563024, -12.2347889, 12.2404490

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 61

### Candidate
type: B, layer: 1, pos: 61

### Candidate
type: B, layer: 1, pos: 54

### Candidate
type: A, layer: 1, pos: 54

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 113

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A1_B1_A2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0118643, upper bound: 9.0072834
time: 3.22 seconds

## Relational analysis of NS_A1_B1_A2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0116985, upper bound: 9.0072572
time: 5.79 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -7.2389727, 5.5630102, -7.2471910, 5.5694709, -12.8084431, 12.8102016
1: -5.8036680, 5.0430980, -5.8097682, 5.0480947, -10.8517628, 10.8528662
2: -7.0746431, 4.2823505, -7.0840812, 4.2869644, -11.3616076, 11.3664322
3: -8.5903301, 4.1791224, -8.5984039, 4.1802235, -12.7705536, 12.7775269
4: -7.7823076, 6.0659208, -7.7904172, 6.0723829, -13.8546906, 13.8563385
5: -6.2085361, 5.2532220, -6.2154026, 5.2584143, -11.4669504, 11.4686241
6: -6.3211918, 6.7112050, -6.3280654, 6.7190976, -13.0402889, 13.0392704
7: -7.9295130, 3.9255381, -7.9400043, 3.9287219, -11.8582344, 11.8655424
8: -7.7804246, 5.7341223, -7.7898188, 5.7405052, -13.5209293, 13.5239410
9: -5.9784865, 6.4232306, -5.9841619, 6.4305439, -12.4090309, 12.4073925

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 61

### Candidate
type: A, layer: 1, pos: 61

### Candidate
type: B, layer: 1, pos: 54

### Candidate
type: A, layer: 1, pos: 54

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of NS_A1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 113

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A1_B1_A2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0118643, upper bound: 9.0072874
time: 2.86 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0116985, upper bound: 9.0072603
time: 5.82 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -7.4367933, 5.7155194, -7.0473199, 5.4154477, -12.8522415, 12.7628393
1: -5.9693127, 5.1764627, -5.6427326, 4.9136367, -10.8829498, 10.8191948
2: -7.2740884, 4.3929806, -6.8826928, 4.1755285, -11.4496174, 11.2756729
3: -8.8348827, 4.2843266, -8.3512917, 4.0744028, -12.9092855, 12.6356182
4: -7.9988441, 6.2275128, -7.5719051, 5.9096155, -13.9084597, 13.7994175
5: -6.3864961, 5.3912411, -6.0357161, 5.1193886, -11.5058842, 11.4269571
6: -6.4918327, 6.8927307, -6.1560755, 6.5356712, -13.0275040, 13.0488062
7: -8.1449518, 4.0345974, -7.7226095, 3.8198938, -11.9648457, 11.7572069
8: -7.9941211, 5.8842411, -7.5738282, 5.5890718, -13.5831928, 13.4580688
9: -6.1449518, 6.5966330, -5.8172183, 6.2563024, -12.4012547, 12.4138508

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 61

### Candidate
type: B, layer: 1, pos: 61

### Candidate
type: B, layer: 1, pos: 54

### Candidate
type: A, layer: 1, pos: 54

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A1_B1_A2_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0118695, upper bound: 9.0076173
time: 4.90 seconds

## Relational analysis of NS_A1_B1_A2_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0117290, upper bound: 9.0072634
time: 5.25 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -7.4367933, 5.7155194, -7.2471910, 5.5694709, -13.0062637, 12.9627104
1: -5.9693127, 5.1764627, -5.8097682, 5.0480947, -11.0174074, 10.9862309
2: -7.2740884, 4.3929806, -7.0840812, 4.2869644, -11.5610523, 11.4770622
3: -8.8348827, 4.2843266, -8.5984039, 4.1802235, -13.0151062, 12.8827305
4: -7.9988441, 6.2275128, -7.7904172, 6.0723829, -14.0712271, 14.0179300
5: -6.3864961, 5.3912411, -6.2154026, 5.2584143, -11.6449108, 11.6066437
6: -6.4918327, 6.8927307, -6.3280654, 6.7190976, -13.2109299, 13.2207966
7: -8.1449518, 4.0345974, -7.9400043, 3.9287219, -12.0736732, 11.9746017
8: -7.9941211, 5.8842411, -7.7898188, 5.7405052, -13.7346268, 13.6740599
9: -6.1449518, 6.5966330, -5.9841619, 6.4305439, -12.5754957, 12.5807953

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 61

### Candidate
type: B, layer: 1, pos: 61

### Candidate
type: B, layer: 1, pos: 54

### Candidate
type: A, layer: 1, pos: 54

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 113

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A1_B1_A2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0118861, upper bound: 9.0072910
time: 4.65 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0117290, upper bound: 9.0072668
time: 3.27 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -7.0309415, 5.4024701, -7.2557211, 5.5762873, -12.6072292, 12.6581917
1: -5.6287084, 4.9021740, -5.8180184, 5.0548301, -10.6835384, 10.7201920
2: -6.8653851, 4.1652822, -7.0923519, 4.2928352, -11.1582203, 11.2576342
3: -8.3310232, 4.0649376, -8.6110897, 4.1888218, -12.5198450, 12.6760273
4: -7.5530319, 5.8955359, -7.8016205, 6.0803370, -13.6333694, 13.6971569
5: -6.0208292, 5.1075163, -6.2237630, 5.2653756, -11.2862053, 11.3312798
6: -6.1413541, 6.5199361, -6.3362565, 6.7273030, -12.8686571, 12.8561926
7: -7.7032814, 3.8099103, -7.9492846, 3.9357767, -11.6390581, 11.7591953
8: -7.5556479, 5.5761518, -7.7990208, 5.7473426, -13.3029900, 13.3751726
9: -5.8027539, 6.2413211, -5.9932928, 6.4385624, -12.2413158, 12.2346134

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 61

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 61

### Candidate
type: B, layer: 1, pos: 54

### Candidate
type: A, layer: 1, pos: 54

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A1_B2_A1_B1_A1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0128461, upper bound: 9.0087474
time: 5.03 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0127146, upper bound: 9.0087194
time: 3.79 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.2305493, 5.5562963, -7.2557211, 5.5762873, -12.8068371, 12.8120174
1: -5.7955317, 5.0364585, -5.8180184, 5.0548301, -10.8503618, 10.8544769
2: -7.0665126, 4.2765837, -7.0923519, 4.2928352, -11.3593483, 11.3689356
3: -8.5778008, 4.1706429, -8.6110897, 4.1888218, -12.7666225, 12.7817326
4: -7.7712717, 6.0580959, -7.8016205, 6.0803370, -13.8516083, 13.8597164
5: -6.2002831, 5.2463655, -6.2237630, 5.2653756, -11.4656582, 11.4701290
6: -6.3131261, 6.7031245, -6.3362565, 6.7273030, -13.0404291, 13.0393810
7: -7.9203949, 3.9186063, -7.9492846, 3.9357767, -11.8561716, 11.8678913
8: -7.7713599, 5.7273903, -7.7990208, 5.7473426, -13.5187025, 13.5264111
9: -5.9694896, 6.4153366, -5.9932928, 6.4385624, -12.4080524, 12.4086294

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 29

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 61

### Candidate
type: A, layer: 1, pos: 61

### Candidate
type: B, layer: 1, pos: 54

### Candidate
type: A, layer: 1, pos: 54

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A1_B2_A1_B1_A2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0128461, upper bound: 9.0087523
time: 5.46 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0127146, upper bound: 9.0087266
time: 5.21 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -7.0309415, 5.4024701, -7.4538527, 5.7290359, -12.7599773, 12.8563232
1: -5.6287084, 4.9021740, -5.9839449, 5.1884160, -10.8171244, 10.8861189
2: -6.8653851, 4.1652822, -7.2921095, 4.4036579, -11.2690430, 11.4573917
3: -8.3310232, 4.0649376, -8.8559771, 4.2942019, -12.6252251, 12.9209146
4: -7.5530319, 5.8955359, -8.0185032, 6.2422180, -13.7952499, 13.9140396
5: -6.0208292, 5.1075163, -6.4020081, 5.4036365, -11.4244652, 11.5095243
6: -6.1413541, 6.5199361, -6.5071898, 6.9091001, -13.0504541, 13.0271263
7: -7.7032814, 3.8099103, -8.1650715, 4.0451040, -11.7483854, 11.9749813
8: -7.5556479, 5.5761518, -8.0130463, 5.8977056, -13.4533539, 13.5891981
9: -5.8027539, 6.2413211, -6.1601191, 6.6123052, -12.4150591, 12.4014397

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 29

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 61

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 61

### Candidate
type: B, layer: 1, pos: 54

### Candidate
type: A, layer: 1, pos: 54

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A1_B2_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0128380, upper bound: 9.0087756
time: 4.69 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0127083, upper bound: 9.0087479
time: 4.23 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7.2305493, 5.5562963, -7.4538527, 5.7290359, -12.9595852, 13.0101490
1: -5.7955317, 5.0364585, -5.9839449, 5.1884160, -10.9839478, 11.0204029
2: -7.0665126, 4.2765837, -7.2921095, 4.4036579, -11.4701710, 11.5686932
3: -8.5778008, 4.1706429, -8.8559771, 4.2942019, -12.8720026, 13.0266199
4: -7.7712717, 6.0580959, -8.0185032, 6.2422180, -14.0134897, 14.0765991
5: -6.2002831, 5.2463655, -6.4020081, 5.4036365, -11.6039200, 11.6483736
6: -6.3131261, 6.7031245, -6.5071898, 6.9091001, -13.2222261, 13.2103138
7: -7.9203949, 3.9186063, -8.1650715, 4.0451040, -11.9654989, 12.0836773
8: -7.7713599, 5.7273903, -8.0130463, 5.8977056, -13.6690655, 13.7404366
9: -5.9694896, 6.4153366, -6.1601191, 6.6123052, -12.5817947, 12.5754557

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 61

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 61

### Candidate
type: B, layer: 1, pos: 54

### Candidate
type: A, layer: 1, pos: 54

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of NS_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 113

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A1_B2_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0128380, upper bound: 9.0087833
time: 3.89 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0127083, upper bound: 9.0087580
time: 3.37 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7.2389727, 5.5630102, -7.2557211, 5.5762873, -12.8152599, 12.8187313
1: -5.8036680, 5.0430980, -5.8180184, 5.0548301, -10.8584976, 10.8611164
2: -7.0746431, 4.2823505, -7.0923519, 4.2928352, -11.3674784, 11.3747025
3: -8.5903301, 4.1791224, -8.6110897, 4.1888218, -12.7791519, 12.7902126
4: -7.7823076, 6.0659208, -7.8016205, 6.0803370, -13.8626442, 13.8675413
5: -6.2085361, 5.2532220, -6.2237630, 5.2653756, -11.4739113, 11.4769850
6: -6.3211918, 6.7112050, -6.3362565, 6.7273030, -13.0484943, 13.0474615
7: -7.9295130, 3.9255381, -7.9492846, 3.9357767, -11.8652897, 11.8748226
8: -7.7804246, 5.7341223, -7.7990208, 5.7473426, -13.5277672, 13.5331430
9: -5.9784865, 6.4232306, -5.9932928, 6.4385624, -12.4170494, 12.4165230

Time for backsubstitution: 2.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 61

### Candidate
type: A, layer: 1, pos: 61

### Candidate
type: B, layer: 1, pos: 54

### Candidate
type: A, layer: 1, pos: 54

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A1_B2_A2_B1_A1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0127285, upper bound: 9.0087330
time: 7.24 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0126084, upper bound: 9.0087129
time: 5.86 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.4367933, 5.7155194, -7.2557211, 5.5762873, -13.0130806, 12.9712410
1: -5.9693127, 5.1764627, -5.8180184, 5.0548301, -11.0241432, 10.9944811
2: -7.2740884, 4.3929806, -7.0923519, 4.2928352, -11.5669231, 11.4853325
3: -8.8348827, 4.2843266, -8.6110897, 4.1888218, -13.0237045, 12.8954163
4: -7.9988441, 6.2275128, -7.8016205, 6.0803370, -14.0791817, 14.0291328
5: -6.3864961, 5.3912411, -6.2237630, 5.2653756, -11.6518717, 11.6150036
6: -6.4918327, 6.8927307, -6.3362565, 6.7273030, -13.2191353, 13.2289867
7: -8.1449518, 4.0345974, -7.9492846, 3.9357767, -12.0807285, 11.9838820
8: -7.9941211, 5.8842411, -7.7990208, 5.7473426, -13.7414637, 13.6832619
9: -6.1449518, 6.5966330, -5.9932928, 6.4385624, -12.5835142, 12.5899258

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 153

### Candidate
type: B, layer: 1, pos: 61

### Candidate
type: A, layer: 1, pos: 61

### Candidate
type: B, layer: 1, pos: 54

### Candidate
type: A, layer: 1, pos: 54

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 113

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A1_B2_A2_B1_A2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0127285, upper bound: 9.0087356
time: 5.20 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -9.0126084, upper bound: 9.0087175
time: 7.82 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 7.48 + 604.16 = 611.64 seconds
