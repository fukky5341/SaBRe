## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_7.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 175.108430440685


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-43.8347244, 168.1222839, -43.8347244, 168.1222839, -211.9570007, 211.9570007)
1: (-119.0844269, 383.1814270, -119.0844269, 383.1814270, -502.2658691, 502.2658691)
2: (-175.4753113, 326.4086914, -175.4753113, 326.4086914, -501.8840027, 501.8840027)
3: (-100.4854202, 409.6264343, -100.4854202, 409.6264343, -510.1118469, 510.1118469)
4: (-160.2841492, 286.0808105, -160.2841492, 286.0808105, -446.3649597, 446.3649597)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.91 + 2.28 = 3.19 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -175.1171863, upper bound: 175.1171863

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1171027, upper bound: 175.1171352
time: 1.11 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1171357, upper bound: 175.1171357
time: 1.15 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.35 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.35
Output dim: 0, lower bound: -175.1171027, upper bound: 175.1171352
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.35
Output dim: 0, lower bound: -175.1171357, upper bound: 175.1171357

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -41.6298676, 159.6666718, -42.6257401, 163.5326843, -205.1625519, 202.2924194
1: -112.9842224, 363.9167786, -115.7239380, 372.8738098, -485.8580322, 479.6406860
2: -165.6068726, 310.5035706, -169.7462616, 317.8961182, -483.5029907, 480.2498169
3: -95.2639542, 388.9776917, -97.5975037, 398.4593201, -493.7232666, 486.5751648
4: -151.5583496, 272.0770264, -155.3727417, 278.5624695, -430.1208191, 427.4497681

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1170924, upper bound: 175.1171230
time: 0.86 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1170874, upper bound: 175.1171155
time: 1.00 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -46.2759247, 177.6007538, -42.6854820, 163.5179596, -209.7938843, 220.2862396
1: -124.6526871, 406.4731140, -115.6240921, 372.8908081, -497.5434570, 522.0972290
2: -182.5472260, 349.2170715, -170.6734924, 318.1436157, -500.6908569, 519.8905640
3: -105.1909409, 432.5337524, -97.6001663, 398.4719849, -503.6629333, 530.1339111
4: -167.5224762, 305.6654358, -156.0071411, 278.8454285, -446.3678589, 461.6725159

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1171044, upper bound: 175.1171208
time: 2.09 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1171049, upper bound: 175.1171049
time: 0.81 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.84 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.84
Output dim: 0, lower bound: -175.1170924, upper bound: 175.1171230
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.84
Output dim: 0, lower bound: -175.1170874, upper bound: 175.1171155
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.84
Output dim: 0, lower bound: -175.1171044, upper bound: 175.1171208
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.84
Output dim: 0, lower bound: -175.1171049, upper bound: 175.1171049

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -39.7140923, 152.3106689, -42.2715836, 162.1089783, -201.8230743, 194.5822449
1: -107.7208710, 347.4979858, -114.6809082, 370.1795654, -477.9004211, 462.1788635
2: -157.8375854, 296.4686584, -168.5204010, 315.3603210, -473.1978455, 464.9890747
3: -90.8364868, 371.2813721, -96.7410812, 395.5099792, -486.3464661, 468.0224609
4: -144.5034943, 259.7754822, -154.1025848, 276.3519592, -420.8554688, 413.8780518

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1169801, upper bound: 175.1170443
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1170736, upper bound: 175.1170979
time: 0.77 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -41.3451042, 158.5876312, -42.1193199, 161.6174469, -202.9625549, 200.7069244
1: -112.1846466, 361.5449524, -114.2974930, 368.6639099, -480.8484802, 475.8424377
2: -164.3353424, 308.4652710, -167.4888000, 314.2701416, -478.6054688, 475.9540710
3: -94.5898819, 386.4167786, -96.3951492, 393.9158936, -488.5057068, 482.8119202
4: -150.4319611, 270.2799377, -153.3671875, 275.3656616, -425.7976074, 423.6471252

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1169753, upper bound: 175.1170442
time: 1.03 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1170597, upper bound: 175.1170766
time: 0.84 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -44.5973396, 171.0905304, -42.0834770, 161.1865540, -205.7838898, 213.1740112
1: -120.1813049, 391.8242798, -114.0907364, 368.1186829, -488.2999878, 505.9150085
2: -176.2078705, 336.6437683, -168.5117340, 313.7154846, -489.9233093, 505.1554260
3: -101.4384308, 416.9261475, -96.3145676, 393.3295593, -494.7680054, 513.2407227
4: -161.6777191, 294.6588135, -154.0167389, 274.9754944, -436.6531982, 448.6755371

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1170738, upper bound: 175.1170912
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1170973, upper bound: 175.1171159
time: 0.90 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -45.9493599, 176.3783417, -42.1189499, 161.3965149, -207.3458557, 218.4972839
1: -123.7047501, 403.7946167, -114.0076675, 368.2245483, -491.9292908, 517.8022461
2: -181.0139160, 346.9296570, -168.0862579, 314.1636353, -495.1775208, 515.0159302
3: -104.4138870, 429.5872803, -96.2469330, 393.3821411, -497.7960205, 525.8342285
4: -166.1543732, 303.6558838, -153.7140198, 275.3433228, -441.4976807, 457.3699036

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1170740, upper bound: 175.1170748
time: 1.09 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1170995, upper bound: 175.1170995
time: 0.82 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.86 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.86
Output dim: 0, lower bound: -175.1169801, upper bound: 175.1170443
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.86
Output dim: 0, lower bound: -175.1170736, upper bound: 175.1170979
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.86
Output dim: 0, lower bound: -175.1169753, upper bound: 175.1170442
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.86
Output dim: 0, lower bound: -175.1170597, upper bound: 175.1170766
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.86
Output dim: 0, lower bound: -175.1170738, upper bound: 175.1170912
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.86
Output dim: 0, lower bound: -175.1170973, upper bound: 175.1171159
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.86
Output dim: 0, lower bound: -175.1170740, upper bound: 175.1170748
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.86
Output dim: 0, lower bound: -175.1170995, upper bound: 175.1170995

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -37.7975693, 144.9502716, -41.0831413, 157.4714813, -195.2690430, 186.0334167
1: -102.4289551, 330.7656860, -111.3980255, 359.6661377, -462.0950928, 442.1636963
2: -149.9667206, 282.4467468, -163.5583496, 306.5998535, -456.5665894, 446.0050964
3: -86.3961868, 353.0269775, -93.9837875, 384.1833191, -470.5794983, 447.0107727
4: -137.3313141, 247.4204254, -149.6161041, 268.6272888, -405.9585876, 397.0365295

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167220, upper bound: 175.1164977
time: 1.13 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167220, upper bound: 175.1170443
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -38.4719849, 147.4998016, -40.3520660, 154.5737305, -193.0457153, 187.8518524
1: -104.1792374, 336.4184875, -109.4208908, 352.6928711, -456.8720703, 445.8393860
2: -152.5367279, 287.6232605, -161.0077515, 301.1384888, -453.6752319, 448.6310120
3: -87.8961716, 359.0965576, -92.3513641, 376.8256226, -464.7217712, 451.4479065
4: -139.7155304, 251.9138947, -147.2198792, 263.8341980, -403.5497437, 399.1337891

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168202, upper bound: 175.1165530
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1170736, upper bound: 175.1170979
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -39.5175514, 151.5790100, -40.9690819, 157.1623993, -196.6799316, 192.5480652
1: -107.1716309, 345.5554199, -111.1553040, 358.4860229, -465.6576538, 456.7107239
2: -156.9662170, 295.0829163, -162.8801727, 305.8097229, -462.7759399, 457.9630737
3: -90.3867416, 369.0142517, -93.7570877, 382.9800415, -473.3667908, 462.7713318
4: -143.7067108, 258.4901123, -149.1574707, 267.9197083, -411.6264038, 407.6475830

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1169403, upper bound: 175.1170096
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1169712, upper bound: 175.1170404
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -39.9161606, 153.0836792, -40.2319794, 154.2709656, -194.1871033, 193.3156586
1: -108.1097412, 348.9489441, -109.0971375, 351.8427429, -459.9524841, 458.0460815
2: -158.2434235, 298.3682251, -159.9261169, 300.4448853, -458.6882935, 458.2943420
3: -91.2139053, 372.5943604, -92.0482788, 375.8309937, -467.0448914, 464.6426392
4: -144.9270325, 261.3165283, -146.4156494, 263.1989136, -408.1259460, 407.7321472

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1170118, upper bound: 175.1170463
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1169403, upper bound: 175.1170670
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -41.1586609, 158.0476532, -40.2273064, 154.1927185, -195.3513794, 198.2749481
1: -110.7971802, 361.8722839, -108.9699173, 352.2844543, -463.0816040, 470.8421936
2: -162.4481506, 311.6922607, -160.7483368, 300.4543152, -462.9024658, 472.4406128
3: -93.4456253, 384.4915466, -91.9766769, 376.1928406, -469.6384583, 476.4682312
4: -149.1837463, 272.5713501, -147.0068359, 263.3133240, -412.4970398, 419.5781250

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168725, upper bound: 175.1165823
time: 0.97 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168725, upper bound: 175.1170863
time: 1.47 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -43.7564888, 167.8998871, -41.5868187, 159.2695007, -203.0259857, 209.4867096
1: -117.8543472, 384.5304871, -112.7333679, 363.6773682, -481.5316772, 497.2638550
2: -172.7768250, 330.5121155, -166.5602570, 310.0893555, -482.8661804, 497.0723877
3: -99.4823608, 409.0343018, -95.1721039, 388.6181641, -488.1005249, 504.2063904
4: -158.5505676, 289.2726746, -152.2386627, 271.7831726, -430.3337402, 441.5113220

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168764, upper bound: 175.1166081
time: 0.92 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168967, upper bound: 175.1171139
time: 1.01 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -42.5351410, 163.4203186, -40.3994789, 154.8887024, -197.4238129, 203.8197937
1: -114.4013367, 374.0682373, -109.2481079, 353.5305786, -467.9319153, 483.3163147
2: -167.4574127, 322.1746521, -160.7609863, 301.9050903, -469.3624878, 482.9355774
3: -96.5091934, 397.4020081, -92.2327805, 377.5108032, -474.0199890, 489.6347656
4: -153.8069305, 281.7578735, -147.1142273, 264.5539856, -418.3609009, 428.8721008

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1169149, upper bound: 175.1166204
time: 1.13 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1169149, upper bound: 175.1170699
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -45.0648155, 173.0299835, -41.6214447, 159.4831543, -204.5479584, 214.6514130
1: -121.2633057, 396.1497498, -112.6357422, 363.8504028, -485.1137085, 508.7854919
2: -177.3673096, 340.5042114, -166.0574646, 310.5446777, -487.9119873, 506.5616455
3: -102.3756561, 421.3081360, -95.0928726, 388.7024536, -491.0781250, 516.4010010
4: -162.8556976, 298.0074768, -151.8739624, 272.1611328, -435.0168457, 449.8813782

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1170192, upper bound: 175.1169386
time: 0.95 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1170658, upper bound: 175.1170658
time: 0.92 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.84 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -175.1167220, upper bound: 175.1164977
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -175.1167220, upper bound: 175.1170443
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -175.1168202, upper bound: 175.1165530
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -175.1170736, upper bound: 175.1170979
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -175.1169403, upper bound: 175.1170096
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -175.1169712, upper bound: 175.1170404
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -175.1170118, upper bound: 175.1170463
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -175.1169403, upper bound: 175.1170670
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -175.1168725, upper bound: 175.1165823
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -175.1168725, upper bound: 175.1170863
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -175.1168764, upper bound: 175.1166081
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -175.1168967, upper bound: 175.1171139
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -175.1169149, upper bound: 175.1166204
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -175.1169149, upper bound: 175.1170699
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -175.1170192, upper bound: 175.1169386
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -175.1170658, upper bound: 175.1170658

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -37.3336868, 143.1769562, -38.9516487, 149.2217255, -186.5554199, 182.1286011
1: -101.1226730, 326.8026428, -105.5881042, 340.8832397, -442.0058899, 432.3907166
2: -147.8434448, 279.1342773, -154.7031097, 290.3977356, -438.2411804, 433.8373718
3: -85.2924957, 348.6396790, -89.0755463, 363.9582520, -449.2507324, 437.7152100
4: -135.4584198, 244.5013885, -141.5489502, 254.4120178, -389.8704224, 386.0503235

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167173, upper bound: 175.1164848
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167173, upper bound: 175.1164843
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -37.2991829, 143.0704346, -39.9636078, 153.2018585, -190.5010376, 183.0340424
1: -101.0499191, 326.5530090, -108.3029480, 349.9396973, -450.9896240, 434.8559570
2: -147.8565521, 278.9231262, -159.0288086, 298.4886169, -446.3451538, 437.9519043
3: -85.2322769, 348.4194031, -91.3812561, 373.7930298, -459.0252991, 439.8006592
4: -135.4321136, 244.3209534, -145.5057831, 261.5221252, -396.9542236, 389.8267212

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1169707, upper bound: 175.1170096
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167173, upper bound: 175.1170404
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -37.9811020, 145.6269073, -38.1524887, 146.0935516, -184.0746460, 183.7793884
1: -102.8015900, 332.2220154, -103.4273758, 333.3547058, -436.1562805, 435.6493835
2: -150.2801971, 284.1241455, -151.8045502, 284.5172729, -434.7974854, 435.9287109
3: -86.7329865, 354.5370483, -87.2844925, 355.9642334, -442.6972046, 441.8215332
4: -137.7109070, 248.8292847, -138.8513031, 249.2445374, -386.9554443, 387.6805725

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1155658, upper bound: 175.1153688
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1155238, upper bound: 175.1153591
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -37.9948883, 145.7031555, -39.2830353, 150.4736938, -188.4685669, 184.9861908
1: -102.8629227, 332.3689270, -106.4573593, 343.3797302, -446.2426453, 438.8262939
2: -150.5137177, 284.2407227, -156.5425415, 293.3768616, -443.8905640, 440.7832336
3: -86.7834015, 354.7235413, -89.8507309, 366.9051208, -453.6885071, 444.5742798
4: -137.8952179, 248.9404907, -143.2020874, 257.0386047, -394.9338379, 392.1425781

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1170632, upper bound: 175.1170923
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1170183, upper bound: 175.1170915
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -38.0095062, 145.8652039, -37.4363403, 143.8043823, -181.8138885, 183.3015442
1: -103.0020828, 332.6627808, -101.4907532, 328.0069885, -431.0090637, 434.1535339
2: -150.6711731, 284.2171936, -148.7658234, 280.2935791, -430.9647522, 432.9830322
3: -86.8573456, 354.9450684, -85.5874710, 349.7961426, -436.6535034, 440.5324707
4: -138.0183716, 248.9136353, -136.3228149, 245.4205475, -383.4388733, 385.2364502

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1169349, upper bound: 175.1170096
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168845, upper bound: 175.1169565
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -38.9998589, 149.5981598, -40.0359650, 153.5649719, -192.5648041, 189.6341248
1: -105.7304382, 341.0479126, -108.5623245, 350.2343445, -455.9647827, 449.6101379
2: -154.8582153, 291.3396606, -159.0644989, 299.0350037, -453.8932190, 450.4041138
3: -89.1752625, 364.0488892, -91.5760651, 374.0455017, -463.2207642, 455.6249390
4: -141.7977753, 255.1842194, -145.7015533, 261.9437256, -403.7415161, 400.8856812

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1169463, upper bound: 175.1170245
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168953, upper bound: 175.1169651
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -38.3400993, 147.1242828, -36.9360695, 141.8447571, -180.1848602, 184.0603485
1: -103.7915192, 335.4158630, -100.0574188, 323.4989929, -427.2904968, 435.4732666
2: -151.7428284, 287.0041504, -146.7384186, 276.7182922, -428.4610901, 433.7425537
3: -87.5504150, 357.9584961, -84.4186096, 344.9490967, -432.4994507, 442.3771057
4: -139.0334320, 251.3250122, -134.4291992, 242.2759857, -381.3094177, 385.7542114

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1170118, upper bound: 175.1170463
time: 1.25 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1170107, upper bound: 175.1170249
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -39.3870697, 151.0695648, -39.3128853, 150.7490387, -190.1361084, 190.3824463
1: -106.6423950, 344.3552551, -106.5463104, 343.7783813, -450.4207764, 450.9015198
2: -156.0855560, 294.5492249, -156.1607208, 293.7800293, -449.8655396, 450.7099609
3: -89.9756393, 367.6176758, -89.9003830, 367.0377808, -457.0134277, 457.5180664
4: -142.9736023, 257.9438477, -143.0089417, 257.3144836, -400.2880859, 400.9527893

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1169463, upper bound: 175.1170614
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1170245, upper bound: 175.1170427
time: 1.00 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -40.6206894, 156.0122375, -38.1777611, 146.2624359, -186.8830872, 194.1899872
1: -109.2791824, 357.3812866, -103.3556976, 334.2248840, -443.5040588, 460.7369995
2: -159.8728027, 307.9432983, -152.1301270, 284.9149475, -444.7877502, 460.0734253
3: -92.1481400, 379.5430603, -87.2275772, 356.7422180, -448.8903503, 466.7706299
4: -146.9095917, 269.2657776, -139.1955566, 249.6709442, -396.5805359, 408.4613342

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167928, upper bound: 175.1165389
time: 1.24 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168491, upper bound: 175.1164938
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168163, upper bound: 175.1164857
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168695, upper bound: 175.1165822
time: 0.89 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168692, upper bound: 175.1165823
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -40.7026596, 156.3293915, -39.1327820, 149.9731140, -190.6757355, 195.4621429
1: -109.5322800, 358.0106201, -105.9561691, 342.7626343, -452.2948914, 463.9667358
2: -160.5655060, 308.4153442, -156.1348724, 292.5069885, -453.0724792, 464.5502319
3: -92.3783340, 380.3092651, -89.4284363, 366.0250244, -458.4033508, 469.7377014
4: -147.4710541, 269.6950989, -142.8304901, 256.3364258, -403.8074341, 412.5255737

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1170588, upper bound: 175.1170819
time: 1.31 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168450, upper bound: 175.1168596
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168336, upper bound: 175.1168442
time: 0.94 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -43.2098694, 165.8496094, -39.5305595, 151.3173523, -194.5272064, 205.3801727
1: -116.3123398, 379.9541626, -107.0965271, 345.6032410, -461.9155884, 487.0506897
2: -170.2328796, 326.7288818, -157.8739777, 294.5639343, -464.7968140, 484.6028442
3: -98.1717453, 404.0102539, -90.4020920, 369.1156311, -467.2873230, 494.4123230
4: -156.2984161, 285.9291992, -144.3671112, 258.1448975, -414.4432983, 430.2963257

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1164956, upper bound: 175.1162255
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168204, upper bound: 175.1165413
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -43.3055344, 166.1815491, -40.5350571, 155.2353058, -198.5408325, 206.7166138
1: -116.6001511, 380.6767578, -109.8386459, 354.5716553, -471.1717529, 490.5153809
2: -170.8227997, 327.2670593, -162.2120819, 302.4551086, -473.2778931, 489.4790955
3: -98.4229813, 404.8785400, -92.7247467, 378.8699036, -477.2928772, 497.6032715
4: -156.8211975, 286.4187622, -148.3065643, 265.0793457, -421.9005432, 434.7253418

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1169380, upper bound: 175.1170194
time: 0.92 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1170678, upper bound: 175.1170922
time: 1.04 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -42.0060730, 161.4341278, -38.3694077, 147.0736389, -189.0796967, 199.8035278
1: -112.9058990, 369.6589966, -103.7105331, 335.6549072, -448.5607910, 473.3695374
2: -164.9131470, 318.5085449, -152.2591248, 286.5675659, -451.4807129, 470.7676697
3: -95.2515869, 392.5566101, -87.5511780, 358.2833862, -453.5349731, 480.1077881
4: -151.5857086, 278.5240784, -139.4087830, 251.0744019, -402.6600952, 417.9328308

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168694, upper bound: 175.1165450
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168448, upper bound: 175.1164937
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168264, upper bound: 175.1164908
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1169149, upper bound: 175.1166204
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1169126, upper bound: 175.1166159
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -42.0711555, 161.6735229, -39.3010941, 150.7105408, -192.7816925, 200.9746094
1: -113.1256943, 370.1412048, -106.2487183, 344.0628662, -457.1885376, 476.3899231
2: -165.5507965, 318.8546448, -156.2535400, 293.9822998, -459.5330811, 475.1081848
3: -95.4297485, 393.1409607, -89.6953583, 367.3506775, -462.7804260, 482.8363037
4: -152.0739594, 278.8431091, -143.0189362, 257.5990601, -409.6730347, 421.8620605

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1170646, upper bound: 175.1170668
time: 0.97 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168502, upper bound: 175.1168462
time: 1.00 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168391, upper bound: 175.1168313
time: 2.34 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -43.8599854, 168.4224396, -39.6822701, 151.9891968, -195.8491821, 208.1047058
1: -117.9515076, 385.6267090, -107.3282471, 346.6579285, -464.6094360, 492.9548950
2: -172.4950409, 331.6882629, -158.3485413, 296.2401123, -468.7351685, 490.0367432
3: -99.6007919, 409.9590454, -90.6392136, 370.1756287, -469.7764282, 500.5982666
4: -158.4185944, 290.2351379, -144.8157959, 259.5718994, -417.9904785, 435.0508728

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1164368, upper bound: 175.1166904
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1170192, upper bound: 175.1169382
time: 0.96 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -43.3036880, 166.2190399, -40.3502388, 154.4721680, -197.7758484, 206.5692749
1: -116.4712372, 380.4677124, -109.1452332, 352.0687256, -468.5399780, 489.6129150
2: -170.4505615, 327.4882202, -161.1306000, 301.3243713, -471.7749329, 488.6188354
3: -98.3171692, 404.5297546, -92.2019424, 376.0369263, -474.3540955, 496.7316589
4: -156.5074005, 286.5339050, -147.3518219, 263.9822693, -420.4896545, 433.8857422

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1165954, upper bound: 175.1168662
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1170655, upper bound: 175.1170655
time: 0.93 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.81 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -175.1167173, upper bound: 175.1164848
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -175.1167173, upper bound: 175.1164843
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -175.1169707, upper bound: 175.1170096
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -175.1167173, upper bound: 175.1170404
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -175.1155658, upper bound: 175.1153688
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -175.1155238, upper bound: 175.1153591
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -175.1170632, upper bound: 175.1170923
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -175.1170183, upper bound: 175.1170915
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -175.1169349, upper bound: 175.1170096
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -175.1168845, upper bound: 175.1169565
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -175.1169463, upper bound: 175.1170245
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -175.1168953, upper bound: 175.1169651
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -175.1170118, upper bound: 175.1170463
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -175.1170107, upper bound: 175.1170249
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -175.1169463, upper bound: 175.1170614
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -175.1170245, upper bound: 175.1170427
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -175.1168695, upper bound: 175.1165822
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -175.1168692, upper bound: 175.1165823
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -175.1168450, upper bound: 175.1168596
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -175.1168336, upper bound: 175.1168442
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -175.1164956, upper bound: 175.1162255
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -175.1168204, upper bound: 175.1165413
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -175.1169380, upper bound: 175.1170194
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -175.1170678, upper bound: 175.1170922
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -175.1169149, upper bound: 175.1166204
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -175.1169126, upper bound: 175.1166159
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -175.1168502, upper bound: 175.1168462
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -175.1168391, upper bound: 175.1168313
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -175.1164368, upper bound: 175.1166904
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -175.1170192, upper bound: 175.1169382
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -175.1165954, upper bound: 175.1168662
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -175.1170655, upper bound: 175.1170655

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -33.7999535, 129.8980255, -37.3734856, 143.2711487, -177.0711060, 167.2714996
1: -91.4436874, 296.6131592, -101.2535095, 327.4429321, -418.8866272, 397.8665771
2: -133.5537567, 253.7022858, -148.1752319, 279.0302124, -412.5839844, 401.8775024
3: -77.0821381, 315.9386597, -85.4013367, 349.3642273, -426.4463501, 401.3399658
4: -122.4710083, 222.0653687, -135.6553497, 244.4258728, -366.8967896, 357.7207031

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167104, upper bound: 175.1164683
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167071, upper bound: 175.1164686
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -36.4396935, 139.7633820, -38.4266548, 147.1969147, -183.6366119, 178.1900330
1: -98.6324234, 319.0221252, -104.1410370, 336.2085266, -434.8409424, 423.1631470
2: -144.1985931, 272.6623230, -152.5983276, 286.5755920, -430.7741699, 425.2606506
3: -83.1961212, 340.1821899, -87.8567352, 358.9611206, -442.1572266, 428.0389404
4: -132.1525269, 238.7846375, -139.6371155, 251.0462341, -383.1987610, 378.4217224

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1162306, upper bound: 175.1159927
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1155239, upper bound: 175.1153234
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -33.7164497, 129.6082458, -38.1818733, 146.4166718, -180.1331024, 167.7901154
1: -91.2394485, 295.9587097, -103.4069214, 334.7036133, -425.9430542, 399.3656311
2: -133.3543549, 253.1260681, -151.3907318, 285.7018433, -419.0561218, 404.5167847
3: -76.9100189, 315.2432251, -87.2155762, 357.1930237, -434.1030273, 402.4587708
4: -122.2740479, 221.5639191, -138.6107330, 250.2529144, -372.5269775, 360.1746216

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167104, upper bound: 175.1169926
time: 1.10 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167071, upper bound: 175.1169855
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -36.4163513, 139.6930695, -39.4592247, 151.2416229, -187.6579742, 179.1522980
1: -98.5853653, 318.8576965, -106.9090424, 345.4411011, -444.0264587, 425.7667236
2: -144.2506561, 272.5285950, -157.0099640, 294.8007812, -439.0513916, 429.5385742
3: -83.1603012, 340.0284729, -90.2119370, 368.9696045, -452.1299133, 430.2404175
4: -132.1629333, 238.6738739, -143.6690216, 258.2723999, -390.4353333, 382.3428650

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1169337, upper bound: 175.1170344
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1155239, upper bound: 175.1170336
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -36.1706009, 138.7153015, -37.3394279, 142.9947968, -179.1654053, 176.0547333
1: -97.9426804, 316.2229614, -101.1826782, 326.2367859, -424.1794434, 417.4055786
2: -143.2150421, 270.5515747, -148.4389648, 278.5725708, -421.7875366, 418.9905396
3: -82.6483612, 337.5273132, -85.4024353, 348.1398621, -430.7882080, 422.9297485
4: -131.2393799, 236.9762115, -135.7930450, 244.0028381, -375.2422180, 372.7692566

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1153290, upper bound: 175.1150750
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1154483, upper bound: 175.1153688
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -35.9848900, 138.0210724, -36.9019165, 141.3260498, -177.3109436, 174.9229889
1: -97.4412155, 314.5784607, -100.0836334, 322.2803040, -419.7215271, 414.6620483
2: -142.4571075, 269.2830811, -146.9597321, 275.1892395, -417.6463623, 416.2427979
3: -82.1928711, 335.7239380, -84.4547272, 344.1403198, -426.3331604, 420.1786194
4: -130.5214081, 235.7981873, -134.3895721, 241.0593719, -371.5807800, 370.1877441

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1154179, upper bound: 175.1150628
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1154179, upper bound: 175.1153591
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -36.2946320, 139.1703796, -38.4039917, 147.1067352, -183.4013672, 177.5743408
1: -98.3402328, 317.2054443, -104.0410690, 335.6510925, -433.9913330, 421.2464600
2: -143.9430542, 271.4165344, -152.8367462, 286.9676819, -430.9107361, 424.2532043
3: -82.9846954, 338.6401367, -87.8161774, 358.4102173, -441.3948975, 426.4562988
4: -131.8804169, 237.7419586, -139.8410339, 251.3777924, -383.2582092, 377.5830078

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1170517, upper bound: 175.1170647
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1170632, upper bound: 175.1170354
time: 0.99 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1170632, upper bound: 175.1170923
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -36.0308075, 138.2172394, -38.0574036, 145.8004608, -181.8312683, 176.2746429
1: -97.5838852, 315.0284424, -103.1833267, 332.5142517, -430.0981445, 418.2117310
2: -142.8335114, 269.6542053, -151.8373566, 284.2172546, -427.0507812, 421.4915771
3: -82.3188171, 336.2359009, -87.0761948, 355.3317261, -437.6504822, 423.3120728
4: -130.8370819, 236.1307373, -138.8626862, 249.0044708, -379.8415527, 374.9934082

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1170183, upper bound: 175.1170292
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1170183, upper bound: 175.1170915
time: 0.95 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -36.2079506, 138.8636780, -36.6367836, 140.6965790, -176.9045105, 175.5004425
1: -98.0888748, 316.6001892, -99.3080597, 320.8547363, -418.9435730, 415.9082642
2: -143.3602295, 271.1432190, -145.5278015, 274.5164185, -417.8766479, 416.6710205
3: -82.7057495, 337.7605286, -83.7458725, 342.1224670, -424.8282166, 421.5063171
4: -131.3705750, 237.4420624, -133.3851318, 240.3425751, -371.7130737, 370.8271484

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1165908, upper bound: 175.1166566
time: 0.94 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1165678, upper bound: 175.1166552
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -39.6489906, 152.2387085, -36.7346992, 141.0080719, -180.6570587, 188.9734039
1: -107.5709305, 347.4971619, -99.5696487, 321.5036316, -429.0744934, 447.0668030
2: -156.9365692, 296.9055176, -145.8372650, 275.3130493, -432.2496338, 442.7427979
3: -90.6311722, 371.0853271, -83.9584427, 342.8890381, -433.5201721, 455.0437012
4: -143.9175415, 260.0737915, -133.6872406, 241.0385132, -384.9560547, 393.7610168

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168845, upper bound: 175.1169550
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168845, upper bound: 175.1169565
time: 1.55 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -37.2087212, 142.6531219, -39.1928940, 150.2875671, -187.4962921, 181.8460083
1: -100.8457794, 325.0900879, -106.2660522, 342.7015991, -443.5473633, 431.3561096
2: -147.6113434, 278.3460388, -155.6622162, 292.9029846, -440.5142822, 434.0082397
3: -85.0507278, 346.9731140, -89.6406479, 365.9665222, -451.0172424, 436.6137085
4: -135.2003174, 243.7762299, -142.6072998, 256.5615845, -391.7619019, 386.3835449

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167590, upper bound: 175.1165647
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1169451, upper bound: 175.1170245
time: 1.14 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -40.8543091, 156.7878723, -39.3914108, 150.9907990, -191.8450623, 196.1792450
1: -110.8448944, 357.8333435, -106.7695084, 344.2757874, -455.1206665, 464.6028442
2: -161.9114075, 305.6817932, -156.3042755, 294.5659485, -456.4773254, 461.9859924
3: -93.4165497, 382.1801758, -90.0610046, 367.6252747, -461.0418091, 472.2411804
4: -148.4485321, 267.7713318, -143.2370605, 257.9888916, -406.4373474, 411.0083618

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167059, upper bound: 175.1165159
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168939, upper bound: 175.1169651
time: 1.01 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -36.5801277, 140.3186798, -36.1607895, 138.8429413, -175.4230652, 176.4794617
1: -98.9904938, 319.7271118, -97.9353714, 316.5456543, -415.5361328, 417.6624451
2: -144.5940247, 274.2633972, -143.6443176, 271.0966492, -415.6906738, 417.9077148
3: -83.4880219, 341.1935425, -82.6291809, 337.5139771, -421.0020142, 423.8227234
4: -132.5439758, 240.1452484, -131.5887909, 237.3457031, -369.8896790, 371.7340393

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1166655, upper bound: 175.1166566
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1166384, upper bound: 175.1166989
time: 0.88 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -39.8862495, 153.0228424, -36.1678505, 138.8003693, -178.6865997, 189.1906738
1: -108.0562057, 349.1730347, -97.9564209, 316.4657288, -424.5219116, 447.1294556
2: -157.5441589, 298.9896851, -143.4462128, 271.2808838, -428.8250427, 442.4358826
3: -91.0809937, 372.9138489, -82.6352386, 337.4514771, -428.5324707, 455.5490723
4: -144.5260315, 261.8319092, -131.5264435, 237.4841614, -382.0101929, 393.3583374

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1166618, upper bound: 175.1166857
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1164951, upper bound: 175.1166711
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -37.6392670, 144.3065643, -38.4694710, 147.4662323, -185.1054688, 182.7760315
1: -101.8692169, 328.7648315, -104.2450562, 336.2276917, -438.0969238, 433.0098877
2: -149.0163422, 281.8899536, -152.7924805, 287.6466370, -436.6629639, 434.6824341
3: -85.9435501, 350.9689941, -87.9602814, 358.9396667, -444.8832092, 438.9292603
4: -136.5485840, 246.8315430, -139.9145050, 251.9337006, -388.4822998, 386.7460327

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1169578, upper bound: 175.1170583
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167590, upper bound: 175.1165969
time: 1.01 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1170243, upper bound: 175.1170609
time: 0.99 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -41.1949043, 157.9053955, -38.6626930, 148.1471710, -189.3420715, 196.5680847
1: -111.5908356, 360.2758789, -104.7456207, 337.7689514, -449.3598022, 465.0214539
2: -163.0214844, 308.3961487, -153.3529205, 289.2736206, -452.2951050, 461.7490845
3: -94.0980530, 384.8529053, -88.3767700, 360.5695190, -454.6675720, 473.2296753
4: -149.5002747, 270.0832825, -140.5412750, 253.3255768, -402.8258667, 410.6245728

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167059, upper bound: 175.1165790
time: 0.94 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168529, upper bound: 175.1170388
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -40.3554077, 154.9889221, -37.9690895, 145.4699249, -185.8253174, 192.9579926
1: -108.6196213, 355.0407410, -102.7953186, 332.4120178, -441.0316467, 457.8360596
2: -159.0299683, 305.6311340, -151.3173828, 283.2855530, -442.3154602, 456.9484558
3: -91.5871277, 377.1596069, -86.7556610, 354.8030701, -446.3901672, 463.9152832
4: -146.0592804, 267.2871704, -138.4466248, 248.2545013, -394.3137817, 405.7337952

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168695, upper bound: 175.1165260
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168695, upper bound: 175.1165267
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -40.3703957, 155.0645294, -38.1398964, 146.1177216, -186.4880981, 193.2044220
1: -108.6176071, 355.1789246, -103.2556686, 333.8877563, -442.5053711, 458.4345703
2: -158.9179688, 306.0523682, -151.9890289, 284.6260986, -443.5440063, 458.0413513
3: -91.5884018, 377.2089539, -87.1435394, 356.3840637, -447.9724731, 464.3524780
4: -146.0298615, 267.6128235, -139.0648041, 249.4179840, -395.4478455, 406.6776123

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168725, upper bound: 175.1165258
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168725, upper bound: 175.1165308
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -38.3857880, 147.3853455, -38.1675491, 146.2105865, -184.5963440, 185.5528717
1: -102.9305573, 337.5205688, -103.1678848, 333.9412537, -436.8718262, 440.6884460
2: -150.8332367, 292.0283203, -152.2531433, 285.7188416, -436.5520630, 444.2814026
3: -86.7989883, 357.9826050, -87.1273575, 356.3305664, -443.1295471, 445.1099548
4: -138.6444550, 255.1664734, -139.3216705, 250.2810974, -388.9255371, 394.4881287

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168448, upper bound: 175.1168275
time: 0.97 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168448, upper bound: 175.1168275
time: 1.50 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -39.2214470, 150.7049408, -38.6567230, 148.1428680, -187.3643188, 189.3616638
1: -105.4583817, 345.1400146, -104.6478348, 338.5601807, -444.0185547, 449.7878418
2: -154.5460205, 297.8360596, -154.1605225, 289.0871582, -443.6331787, 451.9965820
3: -88.9031143, 366.3951111, -88.3157425, 361.5516663, -450.4547729, 454.7108459
4: -141.9588776, 260.3717346, -141.0522308, 253.3261261, -395.2850037, 401.4239502

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168335, upper bound: 175.1168119
time: 0.86 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168335, upper bound: 175.1168119
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -41.2220116, 158.2404633, -38.3653030, 146.7736206, -187.9956360, 196.6057739
1: -110.8322296, 362.5964661, -103.8742905, 335.2906799, -446.1229248, 466.4707642
2: -161.9530029, 312.1605225, -153.0047455, 285.9630127, -447.9160156, 465.1652527
3: -93.5330734, 385.2134399, -87.6958008, 357.9959106, -451.5289917, 472.9092407
4: -148.7958374, 273.0922852, -139.9697571, 250.5680084, -399.3637695, 413.0620422

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1163596, upper bound: 175.1160482
time: 0.93 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1155239, upper bound: 175.1154182
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -41.9909286, 161.1059723, -37.7634926, 144.4314117, -186.4223328, 198.8694611
1: -112.9970093, 368.8203125, -102.2811890, 329.4866333, -442.4836121, 471.1015015
2: -165.2577667, 317.8256531, -150.9597931, 281.4909058, -446.7485962, 468.7854614
3: -95.3777390, 392.1091614, -86.3744202, 351.8387146, -447.2164612, 478.4835815
4: -151.8108063, 278.0441895, -138.0237427, 246.6194153, -398.4302368, 416.0678711

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1163596, upper bound: 175.1165366
time: 1.11 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1156262, upper bound: 175.1154512
time: 1.00 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -41.2883072, 158.4752197, -39.3128586, 150.5014648, -191.7897491, 197.7880554
1: -111.0342941, 363.0786133, -106.4622803, 343.7694702, -454.8037415, 469.5408936
2: -162.5709686, 312.4821777, -157.2379608, 293.4512634, -456.0222168, 469.7200928
3: -93.7299042, 385.8330994, -89.9068527, 367.2128601, -460.9427490, 475.7398682
4: -149.2885437, 273.3957825, -143.7644806, 257.1419678, -406.4305115, 417.1602783

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1163596, upper bound: 175.1170123
time: 0.86 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1169306, upper bound: 175.1170194
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -42.0907135, 161.4757996, -38.7486153, 148.2451630, -190.3358765, 200.2244110
1: -113.2945328, 369.5906372, -104.9622116, 338.2646179, -451.5591431, 474.5527344
2: -166.0133972, 318.4052124, -155.2032318, 289.1927490, -455.2061157, 473.6083679
3: -95.6500702, 393.0458374, -88.6477890, 361.4385071, -457.0885620, 481.6936340
4: -152.4279327, 278.5721130, -141.8663177, 253.3936615, -405.8215942, 420.4384155

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1163596, upper bound: 175.1170913
time: 0.98 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1170111, upper bound: 175.1170904
time: 1.05 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -41.7544327, 160.4559021, -38.1782265, 146.3473511, -188.1017761, 198.6341095
1: -112.3096390, 367.3976440, -103.1985092, 333.9872742, -446.2969055, 470.5960693
2: -164.2126770, 316.2489014, -151.5084991, 285.0550232, -449.2676392, 467.7573853
3: -94.7343445, 390.2889709, -87.1183395, 356.4901428, -451.2244873, 477.4073181
4: -150.8448944, 276.5972900, -138.7154846, 249.7595520, -400.6044312, 415.3127747

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1169149, upper bound: 175.1166204
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1169149, upper bound: 175.1166204
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -41.7620544, 160.5112152, -38.3243408, 146.9025116, -188.6645660, 198.8355560
1: -112.2613754, 367.5142517, -103.5900650, 335.2552185, -447.5166016, 471.1043091
2: -163.9811859, 316.6730347, -152.0875092, 286.2282715, -450.2093201, 468.7605591
3: -94.7043839, 390.2828064, -87.4501038, 357.8591919, -452.5635681, 477.7329102
4: -150.7271576, 276.9189758, -139.2508392, 250.7776184, -401.5047607, 416.1697998

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1169126, upper bound: 175.1166159
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1169126, upper bound: 175.1166159
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -39.7070274, 152.5746002, -38.3417892, 147.0036316, -186.7106018, 190.9163818
1: -106.3295059, 349.3405457, -103.4557800, 335.4825745, -441.8120728, 452.7963257
2: -155.2253265, 302.2374268, -152.0667725, 287.3254395, -442.5507812, 454.3041992
3: -89.7424011, 370.4451599, -87.3801346, 357.7808838, -447.5232849, 457.8252869
4: -142.8120117, 264.0938416, -139.2831879, 251.6545715, -394.4665833, 403.3770142

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168502, upper bound: 175.1168275
time: 1.03 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168050, upper bound: 175.1168090
time: 0.86 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168050, upper bound: 175.1168462
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -40.6032333, 156.1237183, -38.8462982, 148.9866638, -189.5898895, 194.9700165
1: -109.0976868, 357.3743896, -105.0129395, 340.0567322, -449.1544189, 462.3873291
2: -159.5981293, 308.3868408, -154.4445496, 290.7156067, -450.3137207, 462.8313599
3: -92.0141373, 379.3467102, -88.6476898, 363.0546265, -455.0687561, 467.9943848
4: -146.6374359, 269.6187439, -141.3575439, 254.7210541, -401.3584900, 410.9762573

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168391, upper bound: 175.1168119
time: 0.98 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168391, upper bound: 175.1168119
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -41.8658638, 160.7799225, -39.1611595, 150.0268707, -191.8927307, 199.9410858
1: -112.5123062, 368.1307373, -105.8544235, 342.2595215, -454.7717896, 473.9851074
2: -164.0934906, 316.6633911, -155.9069061, 292.6300964, -456.7235413, 472.5703125
3: -94.9891891, 391.2808228, -89.4049149, 365.3068237, -460.2960205, 480.6856995
4: -150.8099670, 277.0325012, -142.6607666, 256.3815613, -407.1914673, 419.6932373

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1164302, upper bound: 175.1166691
time: 0.86 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1154796, upper bound: 175.1157645
time: 1.05 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1154795, upper bound: 175.1157575
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -42.8302002, 164.5373993, -39.2461853, 150.3398132, -193.1700134, 203.7835846
1: -115.1067276, 376.8509521, -106.1370163, 342.9339294, -458.0406494, 482.9879761
2: -168.2047577, 324.2924194, -156.5573883, 293.1078796, -461.3126221, 480.8497925
3: -97.2045975, 400.4845886, -89.6305084, 366.1259766, -463.3305664, 490.1151123
4: -154.5283508, 283.7428284, -143.1891937, 256.8186035, -411.3469543, 426.9320068

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1164302, upper bound: 175.1168439
time: 0.94 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1169924, upper bound: 175.1169362
time: 0.90 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1170192, upper bound: 175.1169348
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -41.2620430, 158.4145203, -39.7816658, 152.3277588, -193.5897980, 198.1961670
1: -110.9108276, 362.6698914, -107.5430756, 347.2546082, -458.1654358, 470.2129517
2: -161.7644348, 312.1955261, -158.5061493, 297.3536072, -459.1180420, 470.7016602
3: -93.6009521, 385.4635620, -90.8410110, 370.7290955, -464.3300476, 476.3045654
4: -148.6615143, 273.0979004, -145.0267181, 260.4750977, -409.1365967, 418.1245422

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1165954, upper bound: 175.1168661
time: 0.99 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1165749, upper bound: 175.1168655
time: 1.46 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -42.2605515, 162.2823486, -39.9223747, 152.8484955, -195.1090393, 202.2046661
1: -113.5758820, 371.5715332, -107.9787750, 348.3988647, -461.9747314, 479.5502930
2: -166.1029053, 319.9794006, -159.3867493, 298.2351379, -464.3380432, 479.3661499
3: -95.8814163, 394.9433899, -91.2190628, 372.0652466, -467.9466553, 486.1624146
4: -152.5702972, 279.9421692, -145.7666168, 261.2694092, -413.8397217, 425.7088013

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1170613, upper bound: 175.1170377
time: 0.94 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1170353, upper bound: 175.1170353
time: 1.01 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.04 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1167104, upper bound: 175.1164683
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1167071, upper bound: 175.1164686
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1162306, upper bound: 175.1159927
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1155239, upper bound: 175.1153234
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1167104, upper bound: 175.1169926
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1167071, upper bound: 175.1169855
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1169337, upper bound: 175.1170344
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1155239, upper bound: 175.1170336
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1153290, upper bound: 175.1150750
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1154483, upper bound: 175.1153688
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1154179, upper bound: 175.1150628
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1154179, upper bound: 175.1153591
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1170632, upper bound: 175.1170354
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1170632, upper bound: 175.1170923
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1170183, upper bound: 175.1170292
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1170183, upper bound: 175.1170915
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1165908, upper bound: 175.1166566
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1165678, upper bound: 175.1166552
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1168845, upper bound: 175.1169550
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1168845, upper bound: 175.1169565
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1167590, upper bound: 175.1165647
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1169451, upper bound: 175.1170245
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1167059, upper bound: 175.1165159
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1168939, upper bound: 175.1169651
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1166655, upper bound: 175.1166566
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1166384, upper bound: 175.1166989
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1166618, upper bound: 175.1166857
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1164951, upper bound: 175.1166711
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1167590, upper bound: 175.1165969
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1170243, upper bound: 175.1170609
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1167059, upper bound: 175.1165790
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1168529, upper bound: 175.1170388
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1168695, upper bound: 175.1165260
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1168695, upper bound: 175.1165267
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1168725, upper bound: 175.1165258
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1168725, upper bound: 175.1165308
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1168448, upper bound: 175.1168275
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1168448, upper bound: 175.1168275
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1168335, upper bound: 175.1168119
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1168335, upper bound: 175.1168119
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1163596, upper bound: 175.1160482
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1155239, upper bound: 175.1154182
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1163596, upper bound: 175.1165366
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1156262, upper bound: 175.1154512
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1163596, upper bound: 175.1170123
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1169306, upper bound: 175.1170194
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1163596, upper bound: 175.1170913
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1170111, upper bound: 175.1170904
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1169149, upper bound: 175.1166204
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1169149, upper bound: 175.1166204
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1169126, upper bound: 175.1166159
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1169126, upper bound: 175.1166159
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1168050, upper bound: 175.1168090
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1168050, upper bound: 175.1168462
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1168391, upper bound: 175.1168119
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1168391, upper bound: 175.1168119
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1154796, upper bound: 175.1157645
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1154795, upper bound: 175.1157575
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1169924, upper bound: 175.1169362
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1170192, upper bound: 175.1169348
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1165954, upper bound: 175.1168661
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1165749, upper bound: 175.1168655
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1170613, upper bound: 175.1170377
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.04
Output dim: 0, lower bound: -175.1170353, upper bound: 175.1170353

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -32.9951477, 126.7792282, -35.5126915, 135.9914246, -168.9865417, 162.2919159
1: -89.2409134, 289.4567871, -96.1867294, 310.6803284, -399.9212341, 385.6435242
2: -130.2816467, 247.8851166, -140.6861877, 265.4170532, -395.6986694, 388.5712891
3: -75.2156906, 308.2435913, -81.1230774, 331.5348816, -406.7505493, 389.3666687
4: -119.5012894, 216.9555664, -128.8334198, 232.4747925, -351.9760742, 345.7889404

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167069, upper bound: 175.1164683
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167069, upper bound: 175.1164683
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -33.1359978, 127.2429352, -39.3239479, 150.9802399, -184.1162262, 166.5668793
1: -89.6328812, 290.4166260, -106.6334686, 345.3109131, -434.9437866, 397.0501099
2: -130.7959747, 248.9671326, -155.6900482, 294.2875977, -425.0835571, 404.6571655
3: -75.5567245, 309.3519592, -89.8998413, 368.3200684, -443.8768005, 399.2518005
4: -119.9894943, 217.8999481, -142.7441101, 257.7904968, -377.7799988, 360.6440430

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1163503, upper bound: 175.1162464
time: 1.12 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1163503, upper bound: 175.1164686
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -35.7260284, 137.0282288, -36.7966156, 140.9099579, -176.6359863, 173.8248444
1: -96.6825256, 312.6368713, -99.7406616, 321.6556702, -418.3381958, 412.3775024
2: -141.2993469, 267.4129639, -146.1409302, 274.4174194, -415.7167358, 413.5538940
3: -81.5469360, 333.3410645, -84.1769257, 343.4587402, -425.0056763, 417.5179749
4: -129.5047607, 234.1501312, -133.7755280, 240.4239655, -369.9287109, 367.9255981

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1148942, upper bound: 175.1149827
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1148942, upper bound: 175.1159927
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -35.2023926, 135.0491943, -36.3693047, 139.3465424, -174.5489197, 171.4184875
1: -95.3124466, 308.0305786, -98.6382065, 317.9453430, -413.2577820, 406.6687927
2: -139.3536224, 263.4505615, -144.6324463, 271.2599792, -410.6135559, 408.0830078
3: -80.3871460, 328.4285583, -83.2024307, 339.4797363, -419.8668518, 411.6309814
4: -127.6994019, 230.6934052, -132.3260803, 237.6023712, -365.3017273, 363.0194702

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1148552, upper bound: 175.1148811
time: 1.97 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1148552, upper bound: 175.1153234
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -32.9205780, 126.5262527, -36.3879776, 139.4481964, -172.3687592, 162.9142151
1: -89.0679169, 288.8736572, -98.5107574, 318.5699158, -407.6377563, 387.3843994
2: -130.1717224, 247.3808289, -144.4544830, 272.5028381, -402.6745605, 391.8353271
3: -75.0691986, 307.6295166, -83.0911713, 340.1125488, -415.1817322, 390.7207031
4: -119.3647766, 216.5195312, -132.2630768, 238.6971130, -358.0618896, 348.7825928

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1162547, upper bound: 175.1164121
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1166149, upper bound: 175.1166275
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -33.0048752, 126.7507324, -40.1500969, 154.1395416, -187.1444092, 166.9008026
1: -89.2879333, 289.3283691, -108.8757095, 352.5506897, -441.8385925, 398.2040405
2: -130.3526306, 248.0447083, -159.1795502, 300.7296143, -431.0822449, 407.2242432
3: -75.2574463, 308.2005310, -91.7882080, 376.3325500, -451.5899963, 399.9887390
4: -119.5759430, 217.0919647, -145.8563538, 263.4863586, -383.0622253, 362.9483032

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1169398, upper bound: 175.1169855
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167069, upper bound: 175.1169855
time: 1.11 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -35.6780090, 136.8497925, -37.6394043, 144.2551727, -179.9331818, 174.4891663
1: -96.5550232, 312.2499390, -102.0890808, 329.1899414, -425.7449646, 414.3389893
2: -141.1907959, 267.0867004, -149.9693298, 281.1131287, -422.3038940, 417.0560303
3: -81.4492416, 332.9382935, -86.1529770, 351.7493896, -433.1986084, 419.0912476
4: -129.3768921, 233.8742523, -137.2393341, 246.3315125, -375.7083740, 371.1135864

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1148942, upper bound: 175.1153383
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1148942, upper bound: 175.1170146
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -35.1991119, 135.0527191, -37.4392776, 143.5236664, -178.7227631, 172.4920044
1: -95.3193970, 308.0403748, -101.5109634, 327.4552612, -422.7746277, 409.5513000
2: -139.5074768, 263.4585266, -149.2929230, 279.6669312, -419.1744080, 412.7514038
3: -80.3982544, 328.4761963, -85.6533585, 349.8189087, -430.2171631, 414.1295166
4: -127.8076172, 230.7077179, -136.5551147, 244.9964752, -372.8040771, 367.2627563

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1154983, upper bound: 175.1159314
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1166711, upper bound: 175.1167361
time: 1.05 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -36.1706009, 138.7153015, -36.5312538, 139.9717407, -176.1423340, 175.2465363
1: -97.9426804, 316.2229614, -98.9049606, 319.3875732, -417.3302612, 415.1279297
2: -143.2150421, 270.5515747, -144.7118530, 272.6913147, -415.9063110, 415.2634277
3: -82.6483612, 337.5273132, -83.4559860, 340.7066956, -423.3550415, 420.9833069
4: -131.2393799, 236.9762115, -132.5408478, 238.8109436, -370.0503235, 369.5170593

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1151374, upper bound: 175.1150565
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1151374, upper bound: 175.1150339
time: 1.11 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -36.1706009, 138.7153015, -40.6093445, 155.7726135, -191.9432068, 179.3246307
1: -97.9426804, 316.2229614, -109.2502747, 356.7492371, -454.6918640, 425.4732361
2: -143.2150421, 270.5515747, -159.7950745, 307.0263367, -450.2413330, 430.3466492
3: -82.6483612, 337.5273132, -92.2248611, 379.0465088, -461.6948853, 429.7521667
4: -131.2393799, 236.9762115, -146.7806244, 268.5548706, -399.7942505, 383.7568359

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1151374, upper bound: 175.1153333
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1151374, upper bound: 175.1153688
time: 0.98 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -35.9848900, 138.0210724, -36.0009995, 137.9516907, -173.9365845, 174.0220490
1: -97.4412155, 314.5784607, -97.5397568, 314.6763306, -412.1175232, 412.1182251
2: -142.4571075, 269.2830811, -142.7550049, 268.6336670, -411.0907593, 412.0380859
3: -82.1928711, 335.7239380, -82.2812119, 335.7701721, -417.9630127, 418.0050964
4: -130.5214081, 235.7981873, -130.7212524, 235.2859955, -365.8074036, 366.5193787

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1146741, upper bound: 175.1146435
time: 1.37 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1146741, upper bound: 175.1150628
time: 1.24 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -35.9848900, 138.0210724, -40.3857193, 154.9086761, -190.8935699, 178.4067993
1: -97.4412155, 314.5784607, -108.7855988, 354.5792236, -452.0204468, 423.3640747
2: -142.4571075, 269.2830811, -159.3031616, 305.0980835, -447.5551758, 428.5862427
3: -82.1928711, 335.7239380, -91.8048553, 376.9229736, -459.1158142, 427.5287781
4: -130.5214081, 235.7981873, -146.2331085, 266.8970947, -397.4185181, 382.0311890

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1146741, upper bound: 175.1148664
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1146741, upper bound: 175.1153591
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -36.2946320, 139.1703796, -37.4403534, 143.4855194, -179.7801514, 176.6107330
1: -98.3402328, 317.2054443, -101.3127899, 327.6740417, -426.0142517, 418.5181580
2: -143.9430542, 271.4165344, -148.1077576, 280.1726685, -424.1157227, 419.5242920
3: -82.9846954, 338.6401367, -85.4826508, 349.4461365, -432.4308472, 424.1228027
4: -131.8804169, 237.7419586, -135.7063751, 245.3419647, -377.2223816, 373.4483032

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167915, upper bound: 175.1167722
time: 1.71 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167761, upper bound: 175.1167798
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -36.2946320, 139.1703796, -41.8344917, 160.4760284, -196.7706604, 181.0048676
1: -98.3402328, 317.2054443, -112.4840927, 367.8222046, -466.1623840, 429.6895142
2: -143.9430542, 271.4165344, -164.7394257, 316.7843628, -460.7273865, 436.1559448
3: -82.9846954, 338.6401367, -94.9981384, 390.7643738, -473.7490845, 433.6382446
4: -131.8804169, 237.7419586, -151.3055573, 277.0037231, -408.8841553, 389.0475159

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167915, upper bound: 175.1167941
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167761, upper bound: 175.1168155
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -36.0308075, 138.2172394, -37.0267563, 141.9272003, -177.9580078, 175.2439880
1: -97.5838852, 315.0284424, -100.2741089, 323.9787598, -421.5626526, 415.3025513
2: -142.8335114, 269.6542053, -146.7607727, 276.9609985, -419.7944641, 416.4149780
3: -82.3188171, 336.2359009, -84.5814362, 345.7127991, -428.0315552, 420.8173218
4: -130.8370819, 236.1307373, -134.3900909, 242.5798798, -373.4169617, 370.5207520

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1163993, upper bound: 175.1166709
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167569, upper bound: 175.1167798
time: 1.21 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -36.0308075, 138.2172394, -41.6274910, 159.6519470, -195.6827393, 179.8447266
1: -97.5838852, 315.0284424, -112.0718231, 365.7055359, -463.2894287, 427.1002808
2: -142.8335114, 269.6542053, -164.2865143, 314.7817688, -457.6152954, 433.9407349
3: -82.3188171, 336.2359009, -94.6115341, 388.7391357, -471.0578918, 430.8474426
4: -130.8370819, 236.1307373, -150.8035278, 275.3705139, -406.2075806, 386.9342651

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1163993, upper bound: 175.1166985
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167569, upper bound: 175.1168156
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -35.2756615, 135.2805786, -34.1938515, 131.2884827, -166.5641022, 169.4744110
1: -95.3828888, 308.3466797, -92.2954636, 299.3458862, -394.7287598, 400.6421509
2: -139.2903900, 264.6799316, -134.8286896, 257.3761902, -396.6665039, 399.5086060
3: -80.4281845, 328.7422485, -77.8260803, 318.7035522, -399.1317444, 406.5683289
4: -127.7305450, 231.6669922, -123.8162537, 225.1470032, -352.8775024, 355.4832153

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1162021, upper bound: 175.1164720
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1162021, upper bound: 175.1166551
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -35.7397728, 137.0924072, -35.2788696, 135.5732880, -171.3130646, 172.3712463
1: -96.8076553, 312.5013733, -95.6131439, 308.9802246, -405.7878723, 408.1145020
2: -141.4937744, 267.7847900, -140.1447906, 264.8298035, -406.3235779, 407.9295654
3: -81.6200562, 333.3444519, -80.6155472, 329.2973022, -410.9173584, 413.9599915
4: -129.6600647, 234.4814606, -128.4572296, 231.7958832, -361.4559326, 362.9386902

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1162021, upper bound: 175.1164720
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1162021, upper bound: 175.1166552
time: 1.45 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -39.6489906, 152.2387085, -35.8552475, 137.7010651, -177.3500519, 188.0939331
1: -107.5709305, 347.4971619, -97.1497650, 313.8818970, -421.4528198, 444.6469116
2: -156.9365692, 296.9055176, -142.0283356, 268.8812561, -425.8178101, 438.9338379
3: -90.6311722, 371.0853271, -81.8773956, 334.7749939, -425.4061279, 452.9626465
4: -143.9175415, 260.0737915, -130.1792450, 235.3978577, -379.3153992, 390.2530212

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1162021, upper bound: 175.1164659
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1164951, upper bound: 175.1165789
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -39.6489906, 152.2387085, -40.3167076, 154.9348755, -194.5838623, 192.5554199
1: -107.5709305, 347.4971619, -108.2735138, 354.7719727, -462.3428650, 455.7706909
2: -156.9365692, 296.9055176, -157.9833832, 306.3742065, -463.3107910, 454.8889160
3: -90.6311722, 371.0853271, -91.3535995, 376.5328369, -467.1639404, 462.4389343
4: -143.9175415, 260.0737915, -145.2843628, 267.8568115, -411.7743225, 405.3581543

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1162021, upper bound: 175.1164659
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1164951, upper bound: 175.1165796
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -36.7370110, 140.8666687, -37.2500534, 142.8118896, -179.5489044, 178.1167297
1: -99.5213547, 321.0576477, -100.9422150, 325.5675354, -425.0888977, 421.9998779
2: -145.4794922, 274.9836731, -147.7370911, 278.0469666, -423.5264587, 422.7207642
3: -83.9294434, 342.5856628, -85.1526184, 347.5041809, -431.4336243, 427.7382812
4: -133.3054962, 240.8143768, -135.3695221, 243.5202637, -376.8257446, 376.1838989

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1163959, upper bound: 175.1164317
time: 1.33 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1163959, upper bound: 175.1165647
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -36.7285881, 140.8463898, -38.0984879, 146.1377869, -182.8663788, 178.9448853
1: -99.5177231, 321.0422974, -103.2648239, 333.3637085, -432.8814392, 424.3071289
2: -145.5519104, 274.9519653, -151.0742188, 285.1129456, -430.6648254, 426.0261841
3: -83.9266739, 342.5711365, -87.0998611, 355.7747192, -439.7013855, 429.6709900
4: -133.3361053, 240.7927551, -138.4051666, 249.7096100, -383.0457153, 379.1979065

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1162081, upper bound: 175.1164734
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1162081, upper bound: 175.1166622
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -40.3000183, 154.6531982, -37.3892822, 143.2813110, -183.5813141, 192.0424652
1: -109.2865067, 353.0278931, -101.2964630, 326.6125183, -435.8989868, 454.3243408
2: -159.4366760, 301.7256775, -148.1443787, 279.2750549, -438.7117310, 449.8699951
3: -92.1030426, 376.9579773, -85.4386978, 348.6017456, -440.7047729, 462.3966675
4: -146.2302856, 264.2812805, -135.7795563, 244.5679016, -390.7981873, 400.0608521

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1166578, upper bound: 175.1164132
time: 1.08 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1164119, upper bound: 175.1164317
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1164119, upper bound: 175.1165159
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -40.4219856, 155.1418457, -38.2568245, 146.6962891, -187.1182709, 193.3986664
1: -109.6516647, 354.1629333, -103.6747818, 334.6162109, -444.2678833, 457.8376770
2: -160.0279236, 302.6409912, -151.4595184, 286.5268250, -446.5547485, 454.1004639
3: -92.4070053, 378.1884460, -87.4382248, 357.1170654, -449.5240784, 465.6266479
4: -146.7681580, 265.0955811, -138.8897858, 250.9207764, -397.6889038, 403.9852905

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1162221, upper bound: 175.1164660
time: 1.17 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1165081, upper bound: 175.1165900
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -35.6959763, 136.9340057, -33.9086800, 130.1971588, -165.8931274, 170.8426819
1: -96.4678345, 312.0391846, -91.5444107, 296.7567749, -393.2246094, 403.5835876
2: -140.7360229, 268.1451111, -133.8152466, 255.2675476, -396.0035400, 401.9603577
3: -81.3445892, 332.7511597, -77.2209244, 316.0628357, -397.4074097, 409.9720459
4: -129.0946503, 234.6981812, -122.8400879, 223.3294525, -352.4241028, 357.5382690

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1162020, upper bound: 175.1166975
time: 0.88 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1162021, upper bound: 175.1166990
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -36.1104507, 138.5394745, -34.6713715, 133.2041626, -169.3146057, 173.2108459
1: -97.7032089, 315.6371765, -93.8871765, 303.4994202, -401.2026367, 409.5243530
2: -142.6990509, 270.9014282, -137.7301331, 260.4262695, -403.1252747, 408.6315613
3: -82.3925934, 336.7723389, -79.1765594, 323.4427795, -405.8353271, 415.9488525
4: -130.8081818, 237.1721039, -126.1579819, 227.9307861, -358.7388916, 363.3300781

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1162020, upper bound: 175.1166975
time: 1.01 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1162021, upper bound: 175.1166989
time: 1.66 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -38.9803848, 149.5009613, -33.8989067, 130.0673218, -169.0476685, 183.3998718
1: -105.4459381, 341.0138245, -91.5303497, 296.4221191, -401.8680420, 432.5440979
2: -153.7857208, 292.6236572, -133.7618561, 255.2563171, -409.0419922, 426.3854675
3: -88.9268417, 363.8070679, -77.2120819, 315.7589111, -404.6857605, 441.0191650
4: -141.1260681, 256.1403503, -122.7996521, 223.3061371, -364.4321899, 378.9399109

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1166594, upper bound: 175.1166857
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1165198, upper bound: 175.1166857
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -39.4145851, 151.2234802, -34.5931549, 132.8314056, -172.2459869, 185.8166351
1: -106.7524948, 345.0058594, -93.6592484, 302.6987915, -409.4512329, 438.6650696
2: -155.6351013, 295.5987244, -137.1844330, 259.9967041, -415.6318054, 432.7831421
3: -89.9774628, 368.4101257, -78.9726028, 322.5997009, -412.5770874, 447.3827209
4: -142.7792664, 258.8344727, -125.7751999, 227.5309448, -370.3102112, 384.6096802

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1162146, upper bound: 175.1166711
time: 0.96 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1166355, upper bound: 175.1166711
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -37.1398621, 142.4188538, -36.5345879, 140.0384521, -177.1783142, 178.9534149
1: -100.4862747, 324.4952087, -98.9177856, 319.2929688, -419.7792053, 423.4129944
2: -146.7600555, 278.3330383, -144.6706696, 272.9661255, -419.7261963, 423.0036926
3: -84.7619324, 346.3241577, -83.4680557, 340.6740417, -425.4359741, 429.7922058
4: -134.5342102, 243.6983643, -132.6010895, 239.0378113, -373.5719910, 376.2993774

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168099, upper bound: 175.1165960
time: 1.02 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168099, upper bound: 175.1165969
time: 1.49 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -37.1755867, 142.5663452, -37.4178085, 143.4947968, -180.6703491, 179.9841614
1: -100.5922546, 324.8448181, -101.3479691, 327.2416992, -427.8339539, 426.1927490
2: -147.0339508, 278.6066589, -148.4953308, 280.1139832, -427.1479492, 427.1019897
3: -84.8611450, 346.7192078, -85.5161591, 349.1996765, -434.0608215, 432.2353516
4: -134.7666779, 243.9473724, -135.9272003, 245.3232574, -380.0899048, 379.8745422

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1166751, upper bound: 175.1167265
time: 1.05 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1166637, upper bound: 175.1167199
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -40.6197433, 155.7190399, -36.6668053, 140.4853363, -181.1050415, 192.3858337
1: -109.9868851, 355.2906494, -99.2705078, 320.2556763, -430.2425537, 454.5611267
2: -160.4667816, 304.2963562, -145.1071320, 274.1306458, -434.5974121, 449.4035034
3: -92.7411270, 379.4510498, -83.7519760, 341.6839294, -434.4250183, 463.2030334
4: -147.2065430, 266.4664612, -133.0269165, 240.0316467, -387.2381592, 399.4933777

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167833, upper bound: 175.1164735
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1162466, upper bound: 175.1162470
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -40.7835960, 156.3408051, -37.5481224, 143.9099884, -184.6935883, 193.8889313
1: -110.4600372, 356.7691956, -101.6807327, 328.2395630, -438.6995850, 458.4499207
2: -161.2465210, 305.4940491, -148.6212463, 281.3424988, -442.5890198, 454.1152954
3: -93.1419373, 381.0641174, -85.7854233, 350.2162170, -443.3581543, 466.8495178
4: -147.9131775, 267.5320740, -136.2319641, 246.3517914, -394.2649536, 403.7640381

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1166718, upper bound: 175.1167031
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1166593, upper bound: 175.1166900
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -40.3554077, 154.9889221, -37.7162819, 144.6971741, -185.0525665, 192.7052002
1: -108.6196213, 355.0407410, -102.1984253, 330.6016235, -439.2212524, 457.2391663
2: -159.0299683, 305.6311340, -149.4149933, 281.4425049, -440.4724426, 455.0461426
3: -91.5871277, 377.1596069, -86.1739883, 352.7956238, -444.3827515, 463.3335876
4: -146.0592804, 267.2871704, -136.8637695, 246.5967255, -392.6560059, 404.1509399

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168662, upper bound: 175.1165205
time: 0.86 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -40.3554077, 154.9889221, -41.2453880, 158.3531036, -198.7084961, 196.2343140
1: -108.6196213, 355.0407410, -110.9172897, 363.1806030, -471.8002319, 465.9580383
2: -159.0299683, 305.6311340, -161.9793701, 311.9396057, -470.9695740, 467.6105042
3: -91.5871277, 377.1596069, -93.5961380, 385.8137207, -477.4008179, 470.7557373
4: -146.0592804, 267.2871704, -148.8351593, 272.9467163, -419.0059814, 416.1223145

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168662, upper bound: 175.1165212
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -40.3703957, 155.0645294, -37.8631897, 145.2568054, -185.6271667, 192.9277191
1: -108.6176071, 355.1789246, -102.5915680, 331.8748474, -440.4924622, 457.7704468
2: -158.9179688, 306.0523682, -149.9756165, 282.6251221, -441.5430298, 456.0279236
3: -91.5884018, 377.2089539, -86.5048294, 354.1524963, -445.7409058, 463.7137756
4: -146.0298615, 267.6128235, -137.3838806, 247.6202698, -393.6501465, 404.9967041

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168687, upper bound: 175.1165204
time: 0.95 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -40.3703957, 155.0645294, -41.4135246, 158.9985809, -199.3689728, 196.4780579
1: -108.6176071, 355.1789246, -111.3725204, 364.6420898, -473.2596741, 466.5514526
2: -158.9179688, 306.0523682, -162.6465302, 313.2785339, -472.1965027, 468.6988831
3: -91.5884018, 377.2089539, -93.9793091, 387.3612366, -478.9496460, 471.1882629
4: -146.0298615, 267.6128235, -149.4471283, 274.1120911, -420.1419678, 417.0599365

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168687, upper bound: 175.1165252
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -38.3857880, 147.3853455, -37.5939178, 144.2182922, -182.6040649, 184.9792633
1: -102.9305573, 337.5205688, -101.6137543, 329.6242981, -432.5548706, 439.1342773
2: -150.8332367, 292.0283203, -148.5288391, 281.7730103, -432.6062317, 440.5571289
3: -86.7989883, 357.9826050, -85.7360764, 351.2991638, -438.0981445, 443.7186890
4: -138.6444550, 255.1664734, -136.1958313, 246.7138824, -385.3583374, 391.3623047

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168422, upper bound: 175.1168188
time: 0.97 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168448, upper bound: 175.1168264
time: 0.96 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167713, upper bound: 175.1167583
time: 0.91 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168448, upper bound: 175.1168275
time: 1.29 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168398, upper bound: 175.1168156
time: 1.28 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -38.3857880, 147.3853455, -41.7324333, 160.2994843, -198.6852570, 189.1177826
1: -102.9305573, 337.5205688, -112.0047226, 367.7602539, -470.6907959, 449.5252380
2: -150.8332367, 292.0283203, -163.8087311, 316.7401123, -467.5733643, 455.8370361
3: -86.7989883, 357.9826050, -94.5551300, 390.3700867, -477.1690674, 452.5377197
4: -138.6444550, 255.1664734, -150.5625763, 276.9587402, -415.6031494, 405.7290039

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168422, upper bound: 175.1168188
time: 0.91 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168448, upper bound: 175.1168264
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167713, upper bound: 175.1167583
time: 0.96 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168448, upper bound: 175.1168275
time: 0.86 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168229, upper bound: 175.1167978
time: 0.90 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -39.2214470, 150.7049408, -38.0170937, 145.8639679, -185.0854187, 188.7220306
1: -105.4583817, 345.1400146, -102.9039612, 333.5158691, -438.9742432, 448.0439758
2: -154.5460205, 297.8360596, -150.1042328, 284.6850281, -439.2310486, 447.9402466
3: -88.9031143, 366.3951111, -86.7552643, 355.8656921, -444.7687988, 453.1503601
4: -141.9588776, 260.3717346, -137.6324615, 249.3796692, -391.3385620, 398.0042114

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168335, upper bound: 175.1168107
time: 0.94 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167590, upper bound: 175.1167433
time: 1.00 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168019, upper bound: 175.1167314
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -39.2214470, 150.7049408, -42.1963539, 162.0361328, -201.2575836, 192.9012909
1: -105.4583817, 345.1400146, -113.4128723, 371.7536621, -477.2120361, 458.5528870
2: -154.5460205, 297.8360596, -165.6609650, 319.8345337, -474.3805542, 463.4970093
3: -88.9031143, 366.3951111, -95.7061081, 394.8792419, -483.7823486, 462.1012268
4: -141.9588776, 260.3717346, -152.2231293, 279.8060303, -421.7648926, 412.5948486

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167590, upper bound: 175.1167433
time: 0.97 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168019, upper bound: 175.1167420
time: 1.01 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -40.3248787, 154.8408813, -36.9965820, 141.5194855, -181.8443604, 191.8374634
1: -108.3458481, 354.7812195, -100.2573166, 323.0946350, -431.4404297, 455.0385437
2: -158.1498871, 305.6858521, -147.6155396, 275.6805725, -433.8304443, 453.3013611
3: -91.4244843, 376.6909180, -84.6724167, 344.9852295, -436.4096680, 461.3633118
4: -145.3729095, 267.3773499, -135.0517120, 241.5821533, -386.9550171, 402.4290466

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1148942, upper bound: 175.1149827
time: 0.93 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1152359, upper bound: 175.1160482
time: 1.85 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -40.0300484, 153.6825256, -36.4326248, 139.3768463, -179.4068909, 190.1151276
1: -107.6442871, 351.9883728, -98.7225266, 318.0418091, -425.6860962, 450.7108765
2: -157.3168640, 303.2486877, -145.6132965, 271.4985352, -428.8153687, 448.8619690
3: -90.8287277, 373.8841858, -83.3396149, 339.5645447, -430.3932800, 457.2237549
4: -144.5288239, 265.2637634, -133.1405640, 237.8630981, -382.3918457, 398.4042358

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1150314, upper bound: 175.1150394
time: 0.92 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1148552, upper bound: 175.1154182
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -41.1339378, 157.8617859, -36.3732834, 139.0978394, -180.2317810, 194.2350616
1: -110.6227341, 361.4148254, -98.5854416, 317.2285156, -427.8512573, 460.0002747
2: -161.6419830, 311.6207886, -145.2725067, 271.1500854, -432.7920227, 456.8932495
3: -93.3657608, 384.0226440, -83.2663040, 338.7052307, -432.0709839, 467.2889404
4: -148.5382996, 272.5646057, -132.8696442, 237.5713806, -386.1096802, 405.4342041

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167752, upper bound: 175.1165287
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1153584, upper bound: 175.1155692
time: 0.99 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1153584, upper bound: 175.1165365
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -40.8059502, 156.5742188, -35.8823738, 137.2411652, -178.0470886, 192.4565887
1: -109.8327789, 358.2450562, -97.2485199, 312.8164062, -422.6491394, 455.4935913
2: -160.6819458, 308.9358826, -143.6598816, 267.4347839, -428.1167297, 452.5957642
3: -92.6902542, 380.8187256, -82.1145554, 333.9486389, -426.6388855, 462.9332886
4: -147.5882568, 270.2272949, -131.3016510, 234.2728882, -381.8611450, 401.5289307

Time for backsubstitution: 1.08 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.19 + 417.32 = 420.51 seconds
