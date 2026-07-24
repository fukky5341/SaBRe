## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 188.29661654332


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-59.5201149, 155.0645905, -59.5201149, 155.0645905, -214.5847015, 214.5847015)
1: (-144.8638458, 230.9529877, -144.8638458, 230.9529877, -375.8167725, 375.8167725)
2: (-97.7719955, 222.6016541, -97.7719955, 222.6016541, -320.3736267, 320.3736267)
3: (-155.9648132, 268.5361023, -155.9648132, 268.5361023, -424.5008850, 424.5008850)
4: (-145.0142365, 255.6766205, -145.0142365, 255.6766205, -400.6908569, 400.6908569)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.84 + 2.12 = 2.96 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -188.3342834, upper bound: 188.3342834

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3300130, upper bound: 188.3328293
time: 1.25 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3342834, upper bound: 188.3342834
time: 0.74 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.06 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.06
Output dim: 0, lower bound: -188.3300130, upper bound: 188.3328293
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.06
Output dim: 0, lower bound: -188.3342834, upper bound: 188.3342834

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -53.3530693, 139.3271179, -56.8082809, 148.0125580, -201.3656006, 196.1354065
1: -128.6982117, 207.5555573, -138.2005615, 220.4711914, -349.1694031, 345.7560120
2: -87.3666611, 199.1090698, -93.3056946, 212.5586548, -299.9253235, 292.4147644
3: -139.7742920, 240.3986511, -148.8231964, 256.5381470, -396.3124390, 389.2218323
4: -130.5633087, 229.1490479, -138.4369507, 244.0570221, -374.6203308, 367.5859985

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3299582, upper bound: 188.3299582
time: 0.74 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3299582, upper bound: 188.3328293
time: 0.72 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -58.5414886, 152.5739136, -58.9514084, 153.6189880, -212.1604767, 211.5253143
1: -142.4417877, 227.2532349, -143.4570923, 228.8044281, -371.2462158, 370.7103271
2: -96.1404495, 219.0275879, -96.8237000, 220.5256195, -316.6660767, 315.8512878
3: -153.3723450, 264.2421570, -154.4591522, 266.0426025, -419.4149475, 418.7012024
4: -142.6441956, 251.5583649, -143.6370239, 253.2858734, -395.9300232, 395.1953735

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3328293, upper bound: 188.3300130
time: 0.82 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3328293, upper bound: 188.3342834
time: 0.76 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.41 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 0, lower bound: -188.3299582, upper bound: 188.3299582
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 0, lower bound: -188.3299582, upper bound: 188.3328293
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 0, lower bound: -188.3328293, upper bound: 188.3300130
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 0, lower bound: -188.3328293, upper bound: 188.3342834

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -53.3530693, 139.3271179, -53.3530693, 139.3271179, -192.6801758, 192.6801758
1: -128.6982117, 207.5555573, -128.6982117, 207.5555573, -336.2537842, 336.2537842
2: -87.3666611, 199.1090698, -87.3666611, 199.1090698, -286.4757385, 286.4757385
3: -139.7742920, 240.3986511, -139.7742920, 240.3986511, -380.1729431, 380.1729431
4: -130.5633087, 229.1490479, -130.5633087, 229.1490479, -359.7123413, 359.7123413

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3298567, upper bound: 188.3299411
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3299582, upper bound: 188.3299582
time: 0.94 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -53.3530693, 139.3271179, -58.5414886, 152.5739136, -205.9269409, 197.8686066
1: -128.6982117, 207.5555573, -142.4417877, 227.2532349, -355.9514465, 349.9973450
2: -87.3666611, 199.1090698, -96.1404495, 219.0275879, -306.3942566, 295.2495117
3: -139.7742920, 240.3986511, -153.3723450, 264.2421570, -404.0163879, 393.7709961
4: -130.5633087, 229.1490479, -142.6441956, 251.5583649, -382.1216736, 371.7932434

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3298567, upper bound: 188.3299411
time: 1.20 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3299582, upper bound: 188.3327207
time: 1.09 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -58.5414886, 152.5739136, -53.3530693, 139.3271179, -197.8686066, 205.9269409
1: -142.4417877, 227.2532349, -128.6982117, 207.5555573, -349.9973450, 355.9514465
2: -96.1404495, 219.0275879, -87.3666611, 199.1090698, -295.2495117, 306.3942566
3: -153.3723450, 264.2421570, -139.7742920, 240.3986511, -393.7709961, 404.0164185
4: -142.6441956, 251.5583649, -130.5633087, 229.1490479, -371.7932434, 382.1216736

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3254909, upper bound: 188.3243398
time: 1.19 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3254928, upper bound: 188.3230445
time: 0.82 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -58.5414886, 152.5739136, -58.5414886, 152.5739136, -211.1153870, 211.1153870
1: -142.4417877, 227.2532349, -142.4417877, 227.2532349, -369.6950073, 369.6950073
2: -96.1404495, 219.0275879, -96.1404495, 219.0275879, -315.1679993, 315.1679993
3: -153.3723450, 264.2421570, -153.3723450, 264.2421570, -417.6145020, 417.6145020
4: -142.6441956, 251.5583649, -142.6441956, 251.5583649, -394.2025452, 394.2025452

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3254909, upper bound: 188.3296825
time: 0.95 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3254928, upper bound: 188.3286822
time: 0.92 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.71 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.71
Output dim: 0, lower bound: -188.3298567, upper bound: 188.3299411
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.71
Output dim: 0, lower bound: -188.3299582, upper bound: 188.3299582
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.71
Output dim: 0, lower bound: -188.3298567, upper bound: 188.3299411
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.71
Output dim: 0, lower bound: -188.3299582, upper bound: 188.3327207
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.71
Output dim: 0, lower bound: -188.3254909, upper bound: 188.3243398
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.71
Output dim: 0, lower bound: -188.3254928, upper bound: 188.3230445
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.71
Output dim: 0, lower bound: -188.3254909, upper bound: 188.3296825
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.71
Output dim: 0, lower bound: -188.3254928, upper bound: 188.3286822

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -51.1813354, 133.8749084, -52.0937157, 136.1645203, -187.3458557, 185.9685822
1: -123.1871643, 199.5406036, -125.5007782, 202.9135132, -326.1006775, 325.0413513
2: -83.7470856, 191.1570892, -85.2661209, 194.4934845, -278.2405701, 276.4232178
3: -134.1554108, 230.9798126, -136.5171967, 234.9356689, -369.0910339, 367.4970093
4: -125.4367523, 220.0620270, -127.5926132, 223.8678436, -349.3045959, 347.6546326

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3291534, upper bound: 188.3289073
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3289047, upper bound: 188.3289168
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -53.2817039, 139.3572388, -52.0931854, 136.1251526, -189.4068451, 191.4503784
1: -128.1620789, 207.6663666, -125.4567261, 202.8227081, -330.9848022, 333.1230774
2: -87.1664886, 198.8413086, -85.2418442, 194.3875580, -281.5540161, 284.0831604
3: -139.5358276, 240.3794708, -136.4146423, 234.8428192, -374.3786316, 376.7941284
4: -130.5832977, 229.0112457, -127.5616531, 223.7945404, -354.3778381, 356.5728760

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3291484, upper bound: 188.3288103
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3289069, upper bound: 188.3289069
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -51.1813354, 133.8749084, -57.3175316, 149.4636688, -200.6450043, 191.1924438
1: -123.1871643, 199.5406036, -139.3203583, 222.6662445, -345.8533936, 338.8608704
2: -83.7470856, 191.1570892, -94.0866623, 214.4548950, -298.2019653, 285.2437439
3: -134.1554108, 230.9798126, -150.1611023, 258.8194275, -392.9747314, 381.1409302
4: -125.4367523, 220.0620270, -139.7473450, 246.3726959, -371.8094482, 359.8093567

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3226949, upper bound: 188.3239313
time: 1.37 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3219737, upper bound: 188.3239479
time: 1.36 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -53.2817039, 139.3572388, -57.2264977, 149.2368774, -202.5185852, 196.5837097
1: -128.1620789, 207.6663666, -139.0106964, 222.3492432, -350.5113220, 346.6770020
2: -87.1664886, 198.8413086, -93.9135742, 214.1139832, -301.2803955, 292.7548828
3: -139.5358276, 240.3794708, -149.8834839, 258.4400024, -397.9758301, 390.2629395
4: -130.5832977, 229.0112457, -139.5434723, 245.9742584, -376.5575562, 368.5547180

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3243398, upper bound: 188.3254909
time: 0.99 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3230445, upper bound: 188.3254928
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -57.2906151, 149.2943268, -52.4784355, 137.0657806, -194.3563843, 201.7727661
1: -139.4646149, 222.3456726, -126.5773163, 204.2015228, -343.6661377, 348.9229736
2: -94.0966492, 214.3430328, -85.9256821, 195.8506317, -289.9472656, 300.2687073
3: -150.0908051, 258.6762695, -137.4854279, 236.5557556, -386.6465454, 396.1616516
4: -139.5511780, 246.1908569, -128.4146729, 225.4370880, -364.9882812, 374.6055298

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3239313, upper bound: 188.3226949
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3239313, upper bound: 188.3243398
time: 1.12 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -54.7164383, 143.3679962, -49.6175079, 129.7665100, -184.4829254, 192.9854736
1: -132.8066711, 213.7103271, -119.5669708, 193.3361359, -326.1427307, 333.2772827
2: -89.6944962, 205.4027405, -81.2017365, 185.2875366, -274.9820251, 286.6044006
3: -143.0580292, 248.4463654, -129.9033203, 223.8731842, -366.9311218, 378.3496094
4: -133.0702820, 236.1494598, -121.3327255, 213.3339844, -346.4041748, 357.4821472

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3239478, upper bound: 188.3221610
time: 0.86 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3239478, upper bound: 188.3230445
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -57.2906151, 149.2943268, -57.8232765, 150.6915588, -207.9821777, 207.1175842
1: -139.4646149, 222.3456726, -140.7354889, 224.4365387, -363.9010925, 363.0811157
2: -94.0966492, 214.3430328, -94.9675522, 216.3425751, -310.4392090, 309.3105774
3: -150.0908051, 258.6762695, -151.4886017, 261.0558167, -411.1466064, 410.1648560
4: -139.5511780, 246.1908569, -140.8667755, 248.4808960, -388.0320740, 387.0576172

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3273657, upper bound: 188.3281634
time: 1.32 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3290693, upper bound: 188.3294262
time: 1.01 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -54.7164383, 143.3679962, -54.6224556, 142.6244812, -197.3409119, 197.9904480
1: -132.8066711, 213.7103271, -132.7207794, 212.4543152, -345.2609253, 346.4310913
2: -89.6944962, 205.4027405, -89.6039352, 204.5819550, -294.2764587, 295.0065918
3: -143.0580292, 248.4463654, -142.9601898, 247.0028839, -390.0608521, 391.4064636
4: -133.0702820, 236.1494598, -132.9965210, 235.0752106, -368.1455078, 369.1459961

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3283639, upper bound: 188.3286305
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3287266, upper bound: 188.3286822
time: 0.70 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.26 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -188.3291534, upper bound: 188.3289073
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -188.3289047, upper bound: 188.3289168
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -188.3291484, upper bound: 188.3288103
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -188.3289069, upper bound: 188.3289069
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -188.3226949, upper bound: 188.3239313
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -188.3219737, upper bound: 188.3239479
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -188.3243398, upper bound: 188.3254909
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -188.3230445, upper bound: 188.3254928
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -188.3239313, upper bound: 188.3226949
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -188.3239313, upper bound: 188.3243398
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -188.3239478, upper bound: 188.3221610
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -188.3239478, upper bound: 188.3230445
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -188.3273657, upper bound: 188.3281634
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -188.3290693, upper bound: 188.3294262
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -188.3283639, upper bound: 188.3286305
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -188.3287266, upper bound: 188.3286822

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -50.1219826, 131.1881104, -50.2017670, 131.3891449, -181.5111237, 181.3898468
1: -120.6621857, 195.5565491, -120.9274368, 195.8220825, -316.4842529, 316.4839783
2: -82.0223007, 187.2983856, -82.1652756, 187.5561218, -269.5784302, 269.4636536
3: -131.3714294, 226.4046478, -131.5711060, 226.7410889, -358.1125183, 357.9757690
4: -122.7996445, 215.6850739, -122.9635315, 215.9946289, -338.7942505, 338.6485901

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3279408, upper bound: 188.3276206
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3291235, upper bound: 188.3287879
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -50.8594513, 133.0670929, -51.4342804, 134.5018463, -185.3612671, 184.5013733
1: -122.4103088, 198.3431244, -123.9084702, 200.4475098, -322.8578186, 322.2515564
2: -83.2112885, 189.9870758, -84.1678391, 192.0854950, -275.2967834, 274.1549072
3: -133.3119354, 229.5905914, -134.7900543, 232.0751648, -365.3870850, 364.3806458
4: -124.6370850, 218.7325134, -125.9557571, 221.1307983, -345.7678833, 344.6882629

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3230444, upper bound: 188.3239388
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3219311, upper bound: 188.3215504
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -52.3335419, 136.9749298, -50.2122879, 131.3852539, -183.7187805, 187.1871948
1: -125.9347000, 204.1279602, -120.9275208, 195.7616882, -321.6963806, 325.0554199
2: -85.6313324, 195.4348907, -82.1626892, 187.5046082, -273.1358643, 277.5975647
3: -137.0489807, 236.3250122, -131.5025940, 226.6881256, -363.7371216, 367.8275757
4: -128.2024078, 225.1318207, -122.9486694, 215.9656525, -344.1680603, 348.0804443

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2689797, upper bound: 188.2822175
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2679797, upper bound: 188.2797996
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -52.9407997, 138.4940796, -51.4242020, 134.4363556, -187.3771515, 189.9182739
1: -127.3365479, 206.3887634, -123.8411102, 200.3212433, -327.6577759, 330.2298584
2: -86.5985489, 197.5935059, -84.1285477, 191.9405975, -278.5391541, 281.7219849
3: -138.6403351, 238.8979187, -134.6655731, 231.9368744, -370.5772095, 373.5634766
4: -129.7382507, 227.5916443, -125.9037781, 221.0151062, -350.7533569, 353.4954224

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3231655, upper bound: 188.3241508
time: 1.58 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3224366, upper bound: 188.3224366
time: 1.26 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -50.3497009, 131.7138824, -56.0843353, 146.2338562, -196.5835571, 187.7981873
1: -121.1964188, 196.3247986, -136.3885193, 217.8372650, -339.0336609, 332.7131958
2: -82.3873672, 188.0375977, -92.0750351, 209.8498535, -292.2372131, 280.1125488
3: -131.9784851, 227.3110657, -146.9262085, 253.3405151, -385.3190002, 374.2372742
4: -123.3868637, 216.5348053, -136.6937103, 241.0925293, -364.4793701, 353.2285156

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3195725, upper bound: 188.3232059
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3198333, upper bound: 188.3198821
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -47.3794250, 124.1691360, -53.5319633, 140.3854065, -187.7648010, 177.7010956
1: -113.8298492, 185.1221008, -129.7988586, 209.3014221, -323.1312256, 314.9209595
2: -77.4363785, 177.1159058, -87.7014084, 201.0024261, -278.4387512, 264.8172913
3: -124.0514145, 214.2090302, -139.9550781, 243.2411804, -367.2925110, 354.1641235
4: -116.0578842, 203.9775085, -130.2556458, 231.1768188, -347.2347107, 334.2331238

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3194345, upper bound: 188.3232535
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3196680, upper bound: 188.3199513
time: 1.23 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -52.5324097, 137.4258118, -55.9532738, 145.9127350, -198.4451447, 193.3790741
1: -126.3275223, 204.8081207, -135.9771423, 217.3710175, -343.6984863, 340.7852173
2: -85.9296951, 196.0542450, -91.8318253, 209.3539276, -295.2835693, 287.8860779
3: -137.5689240, 237.0926514, -146.5433960, 252.7895355, -390.3583374, 383.6360474
4: -128.7497253, 225.8303375, -136.3949432, 240.5277863, -369.2775269, 362.2251892

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2823237, upper bound: 188.2709973
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3242106, upper bound: 188.3254807
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -49.5283470, 129.7709656, -53.3913651, 140.0213318, -189.5496826, 183.1622925
1: -118.9993820, 193.4181213, -129.3540649, 208.7994843, -327.7988586, 322.7721252
2: -80.9782333, 184.9996185, -87.4572830, 200.4773712, -281.4555969, 272.4568176
3: -129.6182709, 223.8295441, -139.5453186, 242.6339722, -372.2522583, 363.3748779
4: -121.3167191, 213.1619568, -129.9570160, 230.5483704, -351.8650818, 343.1189575

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3195876, upper bound: 188.3213523
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2816283, upper bound: 188.2709483
time: 0.94 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3229956, upper bound: 188.3254827
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -56.0843353, 146.2338562, -50.3497009, 131.7138824, -187.7981873, 196.5835571
1: -136.3885193, 217.8372650, -121.1964188, 196.3247986, -332.7131958, 339.0336914
2: -92.0750351, 209.8498535, -82.3873672, 188.0375977, -280.1125488, 292.2372131
3: -146.9262085, 253.3405151, -131.9784851, 227.3110657, -374.2372742, 385.3190002
4: -136.6937103, 241.0925293, -123.3868637, 216.5348053, -353.2285156, 364.4794006

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2618811, upper bound: 188.2517792
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3233477, upper bound: 188.3221105
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -55.9532738, 145.9127350, -52.5324097, 137.4258118, -193.3790741, 198.4451447
1: -135.9771423, 217.3710175, -126.3275223, 204.8081207, -340.7852173, 343.6984863
2: -91.8318253, 209.3539276, -85.9296951, 196.0542450, -287.8860779, 295.2835693
3: -146.5433960, 252.7895355, -137.5689240, 237.0926514, -383.6360474, 390.3583679
4: -136.3949432, 240.5277863, -128.7497253, 225.8303375, -362.2251587, 369.2775269

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3231102, upper bound: 188.3233743
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3254807, upper bound: 188.3242106
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -53.5319633, 140.3854065, -47.3794250, 124.1691360, -177.7010956, 187.7648010
1: -129.7988586, 209.3014221, -113.8298492, 185.1221008, -314.9209290, 323.1312256
2: -87.7014084, 201.0024261, -77.4363785, 177.1159058, -264.8172913, 278.4387512
3: -139.9550781, 243.2411804, -124.0514145, 214.2090302, -354.1641235, 367.2925415
4: -130.2556458, 231.1768188, -116.0578842, 203.9775085, -334.2331238, 347.2347107

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2610685, upper bound: 188.2504445
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3234041, upper bound: 188.3216005
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -53.3913651, 140.0213318, -49.5283470, 129.7709656, -183.1622925, 189.5496826
1: -129.3540649, 208.7994843, -118.9993820, 193.4181213, -322.7721252, 327.7988586
2: -87.4572830, 200.4773712, -80.9782333, 184.9996185, -272.4568176, 281.4555969
3: -139.5453186, 242.6339722, -129.6182709, 223.8295441, -363.3748779, 372.2522583
4: -129.9570160, 230.5483704, -121.3167191, 213.1619568, -343.1189575, 351.8650818

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3254434, upper bound: 188.3229692
time: 0.96 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3251197, upper bound: 188.3229360
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -56.4104614, 147.0464630, -56.8186417, 148.1844635, -204.5949249, 203.8651123
1: -137.3746948, 219.0051422, -138.2913055, 220.7149963, -358.0895691, 357.2964478
2: -92.6664429, 211.1251221, -93.3275528, 212.5744324, -305.2408142, 304.4526672
3: -147.7811279, 254.8359680, -148.8667755, 256.6133118, -404.3944397, 403.7027588
4: -137.3440247, 242.5220032, -138.3827515, 244.3355408, -381.6795654, 380.9047546

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3272624, upper bound: 188.3279666
time: 1.18 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3264531, upper bound: 188.3270571
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -56.8979530, 148.3043365, -57.0068169, 148.6344452, -205.5323944, 205.3111572
1: -138.5182190, 220.8753662, -138.7688446, 221.3829498, -359.9011230, 359.6442261
2: -93.4442902, 212.9103088, -93.6117096, 213.3674622, -306.8117065, 306.5220032
3: -149.0586853, 256.9758301, -149.3426666, 257.5235596, -406.5822144, 406.3184814
4: -138.5802917, 244.5630188, -138.8478699, 245.1001587, -383.6803284, 383.4108887

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3290243, upper bound: 188.3294262
time: 0.84 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3288734, upper bound: 188.3293500
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -53.5319633, 140.3854065, -52.4663048, 137.1622772, -190.6942444, 192.8516693
1: -129.7988586, 209.3014221, -127.2092743, 204.4076691, -334.2065430, 336.5106812
2: -87.7014084, 201.0024261, -85.9784470, 196.5403442, -284.2416687, 286.9808350
3: -139.9550781, 243.2411804, -137.3013153, 237.4833527, -377.4384155, 380.5424500
4: -130.2556458, 231.1768188, -127.9006119, 225.9737396, -356.2293396, 359.0774231

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3265196, upper bound: 188.3255558
time: 1.19 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3264302, upper bound: 188.3266583
time: 2.05 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -53.3913651, 140.0213318, -54.3280373, 142.1357574, -195.5271149, 194.3493652
1: -129.3540649, 208.7994843, -131.5811768, 211.8481140, -341.2021790, 340.3806763
2: -87.4572830, 200.4773712, -89.0162964, 203.5584259, -291.0156250, 289.4936523
3: -139.5453186, 242.6339722, -142.1436310, 246.0709839, -385.6163025, 384.7775879
4: -129.9570160, 230.5483704, -132.4991150, 234.0797729, -364.0368042, 363.0474854

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3279226, upper bound: 188.3271240
time: 0.98 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3265330, upper bound: 188.3268256
time: 1.30 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3285195, upper bound: 188.3285097
time: 1.16 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.71 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.71
Output dim: 0, lower bound: -188.3279408, upper bound: 188.3276206
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.71
Output dim: 0, lower bound: -188.3291235, upper bound: 188.3287879
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.71
Output dim: 0, lower bound: -188.3230444, upper bound: 188.3239388
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.71
Output dim: 0, lower bound: -188.3219311, upper bound: 188.3215504
NS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.71
Output dim: 0, lower bound: -188.2689797, upper bound: 188.2822175
NS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.71
Output dim: 0, lower bound: -188.2679797, upper bound: 188.2797996
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.71
Output dim: 0, lower bound: -188.3231655, upper bound: 188.3241508
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.71
Output dim: 0, lower bound: -188.3224366, upper bound: 188.3224366
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.71
Output dim: 0, lower bound: -188.3195725, upper bound: 188.3232059
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.71
Output dim: 0, lower bound: -188.3198333, upper bound: 188.3198821
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.71
Output dim: 0, lower bound: -188.3194345, upper bound: 188.3232535
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.71
Output dim: 0, lower bound: -188.3196680, upper bound: 188.3199513
NS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.71
Output dim: 0, lower bound: -188.2823237, upper bound: 188.2709973
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.71
Output dim: 0, lower bound: -188.3242106, upper bound: 188.3254807
NS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.71
Output dim: 0, lower bound: -188.2816283, upper bound: 188.2709483
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.71
Output dim: 0, lower bound: -188.3229956, upper bound: 188.3254827
NS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.71
Output dim: 0, lower bound: -188.2618811, upper bound: 188.2517792
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.71
Output dim: 0, lower bound: -188.3233477, upper bound: 188.3221105
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.71
Output dim: 0, lower bound: -188.3231102, upper bound: 188.3233743
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.71
Output dim: 0, lower bound: -188.3254807, upper bound: 188.3242106
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.71
Output dim: 0, lower bound: -188.2610685, upper bound: 188.2504445
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.71
Output dim: 0, lower bound: -188.3234041, upper bound: 188.3216005
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.71
Output dim: 0, lower bound: -188.3254434, upper bound: 188.3229692
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.71
Output dim: 0, lower bound: -188.3251197, upper bound: 188.3229360
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.71
Output dim: 0, lower bound: -188.3272624, upper bound: 188.3279666
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.71
Output dim: 0, lower bound: -188.3264531, upper bound: 188.3270571
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.71
Output dim: 0, lower bound: -188.3290243, upper bound: 188.3294262
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.71
Output dim: 0, lower bound: -188.3288734, upper bound: 188.3293500
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.71
Output dim: 0, lower bound: -188.3265196, upper bound: 188.3255558
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.71
Output dim: 0, lower bound: -188.3264302, upper bound: 188.3266583
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.71
Output dim: 0, lower bound: -188.3265330, upper bound: 188.3268256
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.71
Output dim: 0, lower bound: -188.3285195, upper bound: 188.3285097

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -45.1894073, 117.7344589, -47.0453072, 122.7538986, -167.9432678, 164.7797546
1: -108.9403610, 175.3293457, -113.4354172, 182.8304596, -291.7708130, 288.7647095
2: -74.0016632, 168.1979828, -77.0389481, 175.3179626, -249.3195801, 245.2369232
3: -118.4517975, 203.0234375, -123.2970581, 211.7532349, -330.2050171, 326.3204956
4: -110.6378021, 193.5783081, -115.1691284, 201.8264923, -312.4642334, 308.7474365

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3279408, upper bound: 188.3267092
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3278887, upper bound: 188.3269749
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -51.2181473, 134.0148010, -49.6910934, 130.0820312, -181.3001556, 183.7059021
1: -123.3040695, 199.8567505, -119.6966705, 193.8930359, -317.1971130, 319.5533752
2: -83.8274536, 191.3857269, -81.3403702, 185.7058411, -269.5332947, 272.7260742
3: -134.2456360, 231.3341064, -130.2505035, 224.5136719, -358.7592468, 361.5845947
4: -125.4426880, 220.4072418, -121.7180405, 213.8719025, -339.3145752, 342.1252747

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3283397, upper bound: 188.3243982
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3281026, upper bound: 188.3244600
time: 1.11 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -49.4286537, 129.3451233, -50.5711861, 132.2678986, -181.6965485, 179.9162903
1: -118.9879379, 192.8185730, -121.8249741, 197.1208496, -316.1087952, 314.6434937
2: -80.8736725, 184.6385193, -82.7523193, 188.8645782, -269.7382202, 267.3907776
3: -129.5619049, 223.2851410, -132.5328217, 228.2748413, -357.8367004, 355.8179626
4: -121.1052704, 212.6613464, -123.8304291, 217.4848633, -338.5900879, 336.4917603

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2554443, upper bound: 188.2672028
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3222217, upper bound: 188.3230369
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -47.6430855, 125.3449173, -47.6841660, 124.9139481, -172.5569916, 173.0290833
1: -114.3960876, 187.0176086, -114.7034225, 186.1930084, -300.5890503, 301.7209473
2: -77.8423538, 178.5413361, -77.9582977, 178.2284546, -256.0708008, 256.4995728
3: -124.6327362, 216.4015198, -124.8459473, 215.5048370, -340.1375122, 341.2474365
4: -116.5294724, 205.7654419, -116.6901245, 205.2688141, -321.7982483, 322.4555664

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3210182, upper bound: 188.3170786
time: 1.20 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3191206, upper bound: 188.3174735
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -51.5942612, 134.9956055, -50.5305290, 132.1419678, -183.7362061, 185.5261383
1: -124.0623322, 201.2176971, -121.6595154, 196.9112396, -320.9735413, 322.8771973
2: -84.3859787, 192.5606079, -82.6517487, 188.6336517, -273.0196228, 275.2123413
3: -135.1150665, 232.9567261, -132.3159790, 228.0288696, -363.1439209, 365.2726746
4: -126.4302444, 221.8524323, -123.7115936, 217.2439575, -343.6741943, 345.5640259

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3221660, upper bound: 188.3232758
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3222999, upper bound: 188.3233508
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -49.8476143, 131.1825104, -47.7022934, 124.8945389, -174.7421570, 178.8847961
1: -119.7049179, 195.6788483, -114.7407227, 186.1453705, -305.8502808, 310.4195557
2: -81.4813614, 186.7654266, -77.9959106, 178.1628723, -259.6442261, 264.7612915
3: -130.3535156, 226.4646759, -124.8355713, 215.4586792, -345.8121338, 351.3002319
4: -121.9675903, 215.3497314, -116.7137604, 205.2362671, -327.2038269, 332.0634766

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3220078, upper bound: 188.3224134
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3220897, upper bound: 188.3220897
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -48.5235634, 126.9874496, -55.4498558, 144.5900574, -193.1136017, 182.4373016
1: -116.8901520, 189.3091278, -134.8670502, 215.3916016, -332.2817078, 324.1761780
2: -79.4159088, 181.3325043, -91.0373230, 207.5032501, -286.9191589, 272.3697510
3: -127.1778793, 219.3068390, -145.2611084, 250.5385284, -377.7163696, 364.5679321
4: -118.8373718, 208.8597717, -135.1293030, 238.4021301, -357.2395020, 343.9890442

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3189557, upper bound: 188.3212310
time: 1.46 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3165451, upper bound: 188.3208648
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -49.6719055, 129.9872742, -56.0536880, 146.1560059, -195.8279114, 186.0409241
1: -119.5713120, 193.7634277, -136.3149414, 217.7220001, -337.2932129, 330.0783691
2: -81.2927628, 185.6386566, -92.0255508, 209.7417145, -291.0344849, 277.6642151
3: -130.1766815, 224.4338989, -146.8448334, 253.2109680, -383.3876038, 371.2787476
4: -121.7186356, 213.7645264, -136.6186066, 240.9676971, -362.6863403, 350.3831177

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3196529, upper bound: 188.3196211
time: 0.88 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3168531, upper bound: 188.3191141
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -45.5661812, 119.5226669, -52.9057465, 138.7761383, -184.3423157, 172.4284058
1: -109.5597382, 178.1996002, -128.3019867, 206.9109344, -316.4706726, 306.5015259
2: -74.4854279, 170.5303650, -86.6788635, 198.7115631, -273.1969910, 257.2092285
3: -119.2853394, 206.3222351, -138.3086700, 240.5136566, -359.7990112, 344.6308899
4: -111.5457993, 196.4591217, -128.7068481, 228.5525818, -340.0983887, 325.1659546

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3187140, upper bound: 188.3215459
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3188222, upper bound: 188.3215486
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -46.7239723, 122.4920959, -53.5043755, 140.3158569, -187.0398254, 175.9964752
1: -112.2748489, 182.6340790, -129.7331848, 209.1985168, -321.4733582, 312.3672485
2: -76.3825607, 174.7888184, -87.6570511, 200.9060516, -277.2886047, 262.4458008
3: -122.3239670, 211.4169312, -139.8819733, 243.1256866, -365.4496460, 351.2988892
4: -114.4380875, 201.2913055, -130.1879883, 231.0655518, -345.5036316, 331.4792786

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3191206, upper bound: 188.3170786
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3196680, upper bound: 188.3199513
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -51.8513412, 135.7042694, -55.5583725, 144.9189148, -196.7702637, 191.2626343
1: -124.6833649, 202.2580414, -135.0261383, 215.8959656, -340.5792847, 337.2841797
2: -84.7952271, 193.5675201, -91.1756134, 207.9161530, -292.7113647, 284.7430725
3: -135.7813416, 234.1402435, -145.5054779, 251.0829315, -386.8642578, 379.6457214
4: -127.0578995, 223.0035553, -135.4174042, 238.8942871, -365.9521790, 358.4209595

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3241722, upper bound: 188.3253649
time: 1.06 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3233508, upper bound: 188.3247108
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -48.8916702, 128.1604004, -52.9973717, 139.0300140, -187.9216766, 181.1577454
1: -117.4532547, 191.0415192, -128.3979187, 207.3288574, -324.7820740, 319.4394531
2: -79.9157410, 182.6749268, -86.8006363, 199.0429993, -278.9586792, 269.4755554
3: -127.9471512, 221.0695343, -138.5072327, 240.9291077, -368.8762512, 359.5766907
4: -119.7338562, 210.5146179, -128.9808807, 228.9171295, -348.6509705, 339.4954834

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3228424, upper bound: 188.3253662
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3228063, upper bound: 188.3249195
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -55.5338058, 145.3855438, -49.4593735, 129.4558411, -184.9896393, 194.8449097
1: -135.2084808, 216.7380676, -119.0134277, 192.9904022, -328.1988831, 335.7514954
2: -91.1943207, 208.7875977, -80.9207535, 184.8137817, -276.0080872, 289.7083435
3: -145.4946899, 252.1507874, -129.6167908, 223.4589996, -368.9536743, 381.7675171
4: -135.3057861, 239.8950043, -121.2169037, 212.8242340, -348.1300049, 361.1119080

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3225744, upper bound: 188.3188801
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3190367, upper bound: 188.3190488
time: 1.49 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -54.9503479, 143.4167328, -51.6336594, 135.1620178, -190.1122894, 195.0503845
1: -133.5470886, 213.6667480, -124.2197876, 201.4471588, -334.9942627, 337.8865356
2: -90.2002182, 205.6054382, -84.4753647, 192.8206177, -283.0208435, 290.0808105
3: -143.9253082, 248.3564606, -135.2118530, 233.2406158, -377.1659241, 383.5682983
4: -133.9088440, 236.4013977, -126.4887772, 222.1451416, -356.0539856, 362.8901672

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3159776, upper bound: 188.3203513
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3230971, upper bound: 188.3233409
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -55.1312981, 143.8443298, -52.1830292, 136.5413818, -191.6726837, 196.0273590
1: -133.9994049, 214.3005676, -125.4825363, 203.4978790, -337.4972839, 339.7831116
2: -90.4665375, 206.3615723, -85.3481216, 194.7760315, -285.2425537, 291.7096252
3: -144.3837585, 249.2368164, -136.6509857, 235.5748901, -379.9586182, 385.8877869
4: -134.3602142, 237.1284637, -127.8828430, 224.3755646, -358.7357178, 365.0112915

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3235730, upper bound: 188.3232758
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3247108, upper bound: 188.3233508
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -51.5628853, 135.4306335, -46.5021286, 121.9300003, -173.4928894, 181.9327545
1: -124.9987564, 201.9904022, -111.6885529, 181.8119659, -306.8107300, 313.6789551
2: -84.4701004, 193.9285736, -75.9957733, 173.9217072, -258.3918152, 269.9242859
3: -134.7217560, 234.8582764, -121.7288361, 210.3866882, -345.1083984, 356.5870972
4: -125.4390030, 223.0371399, -113.9132156, 200.3012848, -325.7402954, 336.9503479

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3226652, upper bound: 188.3187772
time: 1.39 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3191650, upper bound: 188.3189298
time: 1.07 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -52.3174667, 137.3102570, -49.1125793, 128.7276306, -181.0451050, 186.4228210
1: -126.6794891, 204.7718658, -117.9548950, 191.8784332, -318.5579224, 322.7267456
2: -85.6606522, 196.4805298, -80.2832565, 183.4797668, -269.1404114, 276.7637634
3: -136.7485046, 237.8640900, -128.5340881, 222.0178528, -358.7663574, 366.3981934
4: -127.4010773, 226.0363922, -120.3336411, 211.4214478, -338.8224487, 346.3700256

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3222241, upper bound: 188.3198078
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2696756, upper bound: 188.2805909
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3244625, upper bound: 188.3219102
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -52.3939285, 137.1157227, -47.8884926, 125.3383179, -177.7322235, 185.0042114
1: -126.6639252, 204.6269379, -114.9879837, 186.8827515, -313.5466919, 319.6148682
2: -85.8049469, 196.2328491, -78.2736511, 178.6516266, -264.4565125, 274.5065002
3: -136.9091339, 237.5489197, -125.3177567, 216.1847992, -353.0938721, 362.8666687
4: -127.7444687, 225.6912537, -117.3560104, 205.8757324, -333.6202087, 343.0472717

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3213484, upper bound: 188.3195512
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3220755, upper bound: 188.3197846
time: 1.00 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3205683, upper bound: 188.3192971
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3243157, upper bound: 188.3220934
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -55.3575287, 144.3605804, -56.4673157, 147.2922821, -202.6497803, 200.8278961
1: -134.7276764, 215.0293579, -137.4043274, 219.3955841, -354.1232605, 352.4336548
2: -90.9071274, 207.1808472, -92.7389908, 211.2594299, -302.1664734, 299.9198303
3: -145.0294800, 250.1179962, -147.9483032, 255.0445251, -400.0740051, 398.0662842
4: -134.8461609, 238.0495300, -137.5519714, 242.8465576, -377.6927185, 375.6015015

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3267267, upper bound: 188.3273478
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3267591, upper bound: 188.3274018
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -55.2703400, 143.7119598, -54.7481804, 142.6708679, -197.9412079, 198.4601440
1: -134.3379822, 214.1601105, -133.1018677, 212.5837555, -346.9216614, 347.2619629
2: -90.7804947, 206.1754608, -89.8891525, 204.6262054, -295.4067078, 296.0645752
3: -144.7924652, 248.9228973, -143.4084320, 247.0451050, -391.8375854, 392.3313293
4: -134.8056183, 236.9275665, -133.4550934, 235.2173157, -370.0228577, 370.3825989

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3246598, upper bound: 188.3243966
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3252156, upper bound: 188.3257467
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -54.8119659, 143.0084686, -55.7936440, 145.5498505, -200.3618011, 198.8020935
1: -133.1986389, 213.0741730, -135.6721954, 216.8377686, -350.0364075, 348.7463684
2: -89.9505768, 205.1392059, -91.5774994, 208.8350983, -298.7855835, 296.7167053
3: -143.5855865, 247.7479858, -146.1576996, 252.1471710, -395.7326660, 393.9056702
4: -133.6397247, 235.7404938, -135.9748383, 239.9575500, -373.5972900, 371.7153320

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3145203, upper bound: 188.3231766
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3280593, upper bound: 188.3283673
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -56.6898384, 148.0390930, -55.6697845, 145.2539062, -201.9437408, 203.7088776
1: -137.5780334, 220.5844574, -135.2828979, 216.4115448, -353.9895325, 355.8672791
2: -92.9968033, 212.1695404, -91.3471298, 208.3815460, -301.3783569, 303.5166016
3: -148.4750824, 256.3724365, -145.7962646, 251.6394348, -400.1145020, 402.1687012
4: -138.2942963, 243.9137421, -135.6918488, 239.4394684, -377.7337341, 379.6055908

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3265536, upper bound: 188.3275902
time: 3.01 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3265617, upper bound: 188.3273242
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -51.9883461, 136.4120178, -51.5267410, 134.7386475, -186.7269897, 187.9387512
1: -126.1791916, 203.4164886, -124.9802628, 200.8270264, -327.0062256, 328.3967590
2: -85.2025986, 195.3245544, -84.4543686, 193.0854340, -278.2880249, 279.7789307
3: -135.9256134, 236.4783783, -134.8485870, 233.3571777, -369.2827759, 371.3269653
4: -126.4130936, 224.7157135, -125.5887985, 222.0161591, -348.4292297, 350.3045044

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2935059, upper bound: 188.2963977
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3250713, upper bound: 188.3246588
time: 1.33 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -53.9932175, 141.5878448, -50.6739197, 132.5898743, -186.5830994, 192.2617645
1: -130.6206360, 211.1714172, -122.7526627, 197.7269440, -328.3475037, 333.9240723
2: -88.4217072, 202.8191223, -83.0354767, 190.0554199, -278.4771118, 285.8545837
3: -140.9156036, 245.4016266, -132.4954224, 229.6801147, -370.5957031, 377.8970032
4: -131.3235474, 233.0463715, -123.5078659, 218.4682465, -349.7918091, 356.5541992

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3125144, upper bound: 188.3203683
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3249592, upper bound: 188.3252298
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -52.4794998, 137.7991943, -53.3349266, 139.6035461, -192.0830383, 191.1341248
1: -127.1653900, 205.5088959, -129.2248077, 208.0927124, -335.2580872, 334.7336426
2: -85.9683685, 197.1579742, -87.4001236, 199.9420929, -285.9103699, 284.5580139
3: -137.1664581, 238.7306824, -139.5370789, 241.7562561, -378.9226685, 378.2677612
4: -127.6924210, 226.9172668, -130.0106506, 229.9554291, -357.6477966, 356.9279175

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3257287, upper bound: 188.3242467
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3257778, upper bound: 188.3261224
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -52.5797653, 137.9796753, -53.9755783, 141.2500763, -193.8298340, 191.9552612
1: -127.3862457, 205.7690430, -130.7269287, 210.5368347, -337.9230957, 336.4959717
2: -86.1051712, 197.5236511, -88.4298706, 202.2778168, -288.3829956, 285.9535217
3: -137.4074860, 239.1216125, -141.2138519, 244.5530701, -381.9605713, 380.3354492
4: -127.9464493, 227.1876221, -131.6265564, 232.6246338, -360.5710754, 358.8141785

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3265816, upper bound: 188.3267247
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3266700, upper bound: 188.3264681
time: 0.78 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.54 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3279408, upper bound: 188.3267092
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3278887, upper bound: 188.3269749
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3283397, upper bound: 188.3243982
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3281026, upper bound: 188.3244600
NS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.2554443, upper bound: 188.2672028
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3222217, upper bound: 188.3230369
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3210182, upper bound: 188.3170786
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3191206, upper bound: 188.3174735
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3221660, upper bound: 188.3232758
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3222999, upper bound: 188.3233508
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3220078, upper bound: 188.3224134
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3220897, upper bound: 188.3220897
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3189557, upper bound: 188.3212310
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3165451, upper bound: 188.3208648
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3196529, upper bound: 188.3196211
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3168531, upper bound: 188.3191141
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3187140, upper bound: 188.3215459
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3188222, upper bound: 188.3215486
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3191206, upper bound: 188.3170786
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3196680, upper bound: 188.3199513
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3241722, upper bound: 188.3253649
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3233508, upper bound: 188.3247108
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3228424, upper bound: 188.3253662
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3228063, upper bound: 188.3249195
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3225744, upper bound: 188.3188801
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3190367, upper bound: 188.3190488
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3159776, upper bound: 188.3203513
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3230971, upper bound: 188.3233409
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3235730, upper bound: 188.3232758
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3247108, upper bound: 188.3233508
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3226652, upper bound: 188.3187772
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3191650, upper bound: 188.3189298
NS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.2696756, upper bound: 188.2805909
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3244625, upper bound: 188.3219102
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3205683, upper bound: 188.3192971
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3243157, upper bound: 188.3220934
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3267267, upper bound: 188.3273478
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3267591, upper bound: 188.3274018
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3246598, upper bound: 188.3243966
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3252156, upper bound: 188.3257467
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3145203, upper bound: 188.3231766
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3280593, upper bound: 188.3283673
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3265536, upper bound: 188.3275902
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3265617, upper bound: 188.3273242
NS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.2935059, upper bound: 188.2963977
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3250713, upper bound: 188.3246588
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3125144, upper bound: 188.3203683
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3249592, upper bound: 188.3252298
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3257287, upper bound: 188.3242467
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3257778, upper bound: 188.3261224
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3265816, upper bound: 188.3267247
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -188.3266700, upper bound: 188.3264681

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -43.3459396, 113.0150833, -42.8757057, 111.7107086, -155.0566254, 155.8907776
1: -104.4603424, 168.2949219, -103.3802948, 166.1156158, -270.5758667, 271.6751709
2: -70.9769211, 161.4123840, -70.1945343, 159.2934723, -230.2703857, 231.6068878
3: -113.6237564, 194.8895111, -112.4541321, 192.2491608, -305.8728943, 307.3436279
4: -106.1515579, 185.7659302, -104.9905548, 183.3545380, -289.5061035, 290.7564087

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3249859, upper bound: 188.3260581
time: 1.14 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3251803, upper bound: 188.3239497
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -43.9576530, 114.6218262, -45.2028580, 118.0804291, -162.0380554, 159.8246613
1: -105.9021072, 170.7265930, -108.8902969, 175.9250793, -281.8271790, 279.6168823
2: -71.9592590, 163.7167206, -73.9808273, 168.6020203, -240.5612793, 237.6975403
3: -115.2133942, 197.6687927, -118.4478912, 203.7157135, -318.9290771, 316.1166992
4: -107.6527939, 188.4322357, -110.7017517, 194.1104126, -301.7631836, 299.1339111

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3250934, upper bound: 188.3263103
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3250934, upper bound: 188.3240404
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -49.3034821, 129.1119537, -44.6080856, 116.3022919, -165.6057739, 173.7200317
1: -118.6340714, 192.5547943, -107.6075058, 172.9641876, -291.5982361, 300.1622620
2: -80.6712570, 184.3114929, -73.0647736, 165.8685760, -246.5398102, 257.3762512
3: -129.2240906, 222.8667297, -117.0025101, 200.2316284, -329.4557190, 339.8692322
4: -120.8018799, 212.2836304, -109.1511688, 190.9412537, -311.7431335, 321.4347534

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3222588, upper bound: 188.3118520
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3222588, upper bound: 188.3222454
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -50.1175041, 131.2333374, -47.7280388, 125.1232224, -175.2407227, 178.9613647
1: -120.5991440, 195.7369232, -114.8770065, 186.5544586, -307.1535950, 310.6139221
2: -81.9983292, 187.3898010, -78.0850296, 178.5841217, -260.5824585, 265.4748230
3: -131.3500214, 226.5468597, -125.0853882, 215.9962921, -347.3463135, 351.6322327
4: -122.7685928, 215.8174133, -116.9503937, 205.6961517, -328.4647522, 332.7678223

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3220140, upper bound: 188.3118520
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3263897, upper bound: 188.3222857
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -48.5422592, 127.0976562, -49.9650650, 131.2760620, -179.8182983, 177.0626984
1: -116.8143387, 189.4976349, -120.5041733, 195.8314667, -312.6457520, 310.0017700
2: -79.4140472, 181.4259644, -81.7866440, 187.5986786, -267.0127258, 263.2125854
3: -127.2109756, 219.4502563, -130.9486084, 226.8549500, -354.0659180, 350.3987732
4: -118.9461212, 208.9669952, -122.3119888, 216.0857239, -335.0318604, 331.2789307

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3198992, upper bound: 188.3222713
time: 1.88 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3201178, upper bound: 188.3191626
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -46.9922295, 123.6838226, -45.8576012, 120.2288208, -167.2210388, 169.5414276
1: -112.8279724, 184.5464172, -110.3980637, 179.2160492, -292.0439758, 294.9444580
2: -76.7759628, 176.1707611, -74.9833374, 171.5861969, -248.3621368, 251.1540833
3: -122.9208603, 213.5745850, -120.0357285, 207.5537109, -330.4744873, 333.6102600
4: -114.9281387, 203.0519714, -112.1442108, 197.6842651, -312.6123962, 315.1961670

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3189187, upper bound: 188.3169385
time: 1.18 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3191661, upper bound: 188.3168856
time: 1.00 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -47.6173401, 125.2794418, -47.0310020, 123.2438965, -170.8612366, 172.3104248
1: -114.3352203, 186.9205170, -113.1517105, 183.7160950, -298.0513306, 300.0722351
2: -77.8011856, 178.4503479, -76.9089661, 175.9094391, -253.7106323, 255.3593140
3: -124.5646973, 216.2925110, -123.1244354, 212.7243652, -337.2890015, 339.4169006
4: -116.4662247, 205.6605988, -115.0782242, 202.5919037, -319.0581360, 320.7388306

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3188654, upper bound: 188.3174735
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3188654, upper bound: 188.3174735
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -51.2099609, 134.0211639, -49.4436378, 129.3868866, -180.5968323, 183.4647980
1: -123.0974579, 199.7785492, -118.9387665, 192.8413086, -315.9387817, 318.7173157
2: -83.7483521, 191.1474609, -80.8467560, 184.6259766, -268.3743286, 271.9941711
3: -134.1206207, 231.2703094, -129.5076141, 223.2431030, -357.3637085, 360.7779236
4: -125.5210037, 220.2347412, -121.1414185, 212.6535950, -338.1745911, 341.3761597

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3221660, upper bound: 188.3232758
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3221660, upper bound: 188.3232758
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -49.8664055, 130.3355865, -50.0585747, 130.4648438, -180.3312531, 180.3941345
1: -119.8037338, 194.3261719, -120.3406830, 194.5589294, -314.3625793, 314.6667786
2: -81.5197449, 185.8464508, -81.8687897, 186.1161041, -267.6358337, 267.7152405
3: -130.5545807, 224.8731537, -131.1050568, 225.0998230, -355.6544189, 355.9781494
4: -122.2783356, 214.1624908, -122.7471848, 214.4076233, -336.6859436, 336.9096375

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3192671, upper bound: 188.3213168
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3214952, upper bound: 188.3225729
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -49.4609680, 130.2113647, -46.5564384, 122.0043259, -171.4653015, 176.7678070
1: -118.7328568, 194.2490692, -111.8665619, 181.8657532, -300.5986023, 306.1156006
2: -80.8337021, 185.3458252, -76.0784531, 173.9603729, -254.7940674, 261.4242859
3: -129.3506317, 224.7813568, -121.8511505, 210.4318542, -339.7824707, 346.6325073
4: -121.0527496, 213.7252502, -114.0014725, 200.4181366, -321.4708862, 327.7267151

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3190405, upper bound: 188.3207923
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3211785, upper bound: 188.3215192
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -48.2308655, 126.8647461, -47.2728806, 123.3950882, -171.6259460, 174.1376038
1: -115.7308960, 189.3433380, -113.5536652, 184.0522919, -299.7831421, 302.8970032
2: -78.8118286, 180.5851898, -77.2760696, 175.9548798, -254.7666473, 257.8612671
3: -126.1160736, 219.0266266, -123.7306137, 212.8966980, -339.0127563, 342.7572327
4: -118.0886383, 208.2559814, -115.8119888, 202.7132263, -320.8018799, 324.0679626

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3190887, upper bound: 188.3204521
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3212797, upper bound: 188.3212797
time: 1.19 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -48.3994637, 126.6667175, -54.9391098, 143.2529449, -191.6523895, 181.6058350
1: -116.5967026, 188.8304443, -133.6507721, 213.3915253, -329.9882202, 322.4812012
2: -79.2130051, 180.8731537, -90.2008438, 205.5853729, -284.7983704, 271.0739441
3: -126.8533401, 218.7566681, -143.9230499, 248.2350311, -375.0883789, 362.6797180
4: -118.5319901, 208.3321228, -133.8715973, 236.1965790, -354.7285156, 342.2037048

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3193381, upper bound: 188.3208008
time: 1.28 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3193381, upper bound: 188.3212310
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -47.5679169, 124.5284424, -55.2735825, 144.1579285, -191.7258301, 179.8020020
1: -114.5641556, 185.6943207, -134.4502716, 214.9351959, -329.4993591, 320.1445618
2: -77.8341064, 177.8239746, -90.7632980, 207.0158386, -284.8498535, 268.5872498
3: -124.6681061, 215.1162262, -144.8197174, 250.0177155, -374.6858215, 359.9358215
4: -116.5230637, 204.8331757, -134.7581024, 237.8355865, -354.3586121, 339.5912170

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3165451, upper bound: 188.3206378
time: 1.19 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3165451, upper bound: 188.3208647
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -49.5287285, 129.6131439, -55.5237541, 144.7656250, -194.2943420, 185.1369019
1: -119.2302475, 193.2040100, -135.0507050, 215.6409149, -334.8710938, 328.2546997
2: -81.0586777, 185.1010895, -91.1572723, 207.7445374, -288.8031921, 276.2583618
3: -129.8018646, 223.7907715, -145.4558716, 250.8132019, -380.6150208, 369.2465820
4: -121.3673019, 213.1477661, -135.3145294, 238.6721954, -360.0394592, 348.4622192

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3196529, upper bound: 188.3194879
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3196529, upper bound: 188.3196211
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -48.7527809, 127.6338196, -55.8574104, 145.6674042, -194.4201813, 183.4911652
1: -117.3348770, 190.3081360, -135.8467407, 217.1806793, -334.5155640, 326.1548462
2: -79.7711258, 182.2841339, -91.7178802, 209.1721802, -288.9432983, 274.0020142
3: -127.7621231, 220.4274139, -146.3488159, 252.5899200, -380.3520508, 366.7762451
4: -119.4927902, 209.9129639, -136.1979523, 240.3055267, -359.7982483, 346.1109009

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2708179, upper bound: 188.2820991
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2589619, upper bound: 188.2596760
time: 1.08 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -45.4316673, 119.1777725, -52.4979935, 137.7220612, -183.1537323, 171.6757660
1: -109.2399750, 177.6849518, -127.3389969, 205.3295135, -314.5694580, 305.0239563
2: -74.2644501, 170.0362244, -86.0116653, 197.1997528, -271.4641418, 256.0478821
3: -118.9335251, 205.7305908, -137.2422180, 238.6958771, -357.6293945, 342.9727783
4: -111.2164459, 195.8921204, -127.7009811, 226.8175507, -338.0339966, 323.5931091

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3188222, upper bound: 188.3215458
time: 1.09 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3188222, upper bound: 188.3215458
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -44.7543068, 117.4817352, -52.4508057, 137.7459717, -182.5002747, 169.9325409
1: -107.6081696, 175.2289276, -127.2731247, 205.6085510, -313.2167053, 302.5020447
2: -73.1449280, 167.6462250, -85.9638062, 197.4566803, -270.6015320, 253.6100311
3: -117.1601715, 202.8830566, -137.1416779, 239.1007996, -356.2609863, 340.0247192
4: -109.5750046, 193.1499329, -127.6188507, 227.0982513, -336.6732483, 320.7687378

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3165451, upper bound: 188.3209351
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3165451, upper bound: 188.3213730
time: 1.29 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -46.7239723, 122.4920959, -51.8500595, 136.0684509, -182.7924194, 174.3421631
1: -112.2748489, 182.6340790, -125.8719330, 202.8921051, -315.1669617, 308.5059814
2: -76.3825607, 174.7888184, -84.9765472, 194.8867645, -271.2693176, 259.7653198
3: -122.3239670, 211.4169312, -135.5384979, 235.9654388, -358.2893982, 346.9554138
4: -114.4380875, 201.2913055, -126.0512695, 224.2078094, -338.6458130, 327.3425903

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3181953, upper bound: 188.3189197
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3190748, upper bound: 188.3188466
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -46.7239723, 122.4920959, -52.9110184, 138.8148956, -185.5388641, 175.4031067
1: -112.2748489, 182.6340790, -128.3170624, 206.9779663, -319.2528076, 310.9511414
2: -76.3825607, 174.7888184, -86.7014160, 198.8250732, -275.2076111, 261.4902039
3: -122.3239670, 211.4169312, -138.3075256, 240.6320648, -362.9560242, 349.7244263
4: -114.4380875, 201.2913055, -128.7320251, 228.6633301, -343.1014099, 330.0233154

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3189050, upper bound: 188.3189197
time: 1.28 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3190748, upper bound: 188.3188466
time: 0.88 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -51.4773636, 134.7557831, -54.5109138, 142.2438507, -193.7212219, 189.2666931
1: -123.7457504, 200.8588867, -132.3990936, 211.9355774, -335.6813354, 333.2579956
2: -84.1737671, 192.1877441, -89.4288025, 203.9870605, -288.1607666, 281.6164551
3: -134.8145752, 232.4951630, -142.7702484, 246.3873749, -381.2019043, 375.2654114
4: -126.1728058, 221.4240112, -132.9319458, 234.4389343, -360.6117554, 354.3558655

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3197707, upper bound: 188.3219156
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2801124, upper bound: 188.2689167
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3229697, upper bound: 188.3243913
time: 1.10 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -50.1172180, 131.0264130, -54.4937286, 141.7694855, -191.8867035, 185.5201416
1: -120.4092941, 195.3442841, -132.1902618, 211.3190002, -331.7282715, 327.5345154
2: -81.9191437, 186.8337097, -89.4221878, 203.2224121, -285.1415100, 276.2558289
3: -131.2086945, 226.0322571, -142.7217560, 245.4734955, -376.6821289, 368.7539673
4: -122.8904572, 215.2957001, -133.0535431, 233.5992126, -356.4896851, 348.3492432

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3197168, upper bound: 188.3216440
time: 1.16 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3194076, upper bound: 188.3204936
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3225729, upper bound: 188.3239720
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -48.4974709, 127.1717224, -51.9264221, 136.3271637, -184.8246155, 179.0981445
1: -116.4623184, 189.5807343, -125.7336426, 203.3132935, -319.7755737, 315.3143921
2: -79.2559891, 181.2347717, -85.0104370, 195.0581512, -274.3141479, 266.2451172
3: -126.9197922, 219.3517609, -135.7186890, 236.1735992, -363.0933838, 355.0704346
4: -118.8011856, 208.8656464, -126.4320831, 224.4192352, -343.2204285, 335.2976990

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3196586, upper bound: 188.3219100
time: 1.35 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2804533, upper bound: 188.2694271
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3218578, upper bound: 188.3243864
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -47.2296562, 123.6734619, -52.0178909, 136.1718140, -183.4014740, 175.6913452
1: -113.3901062, 184.4282684, -125.7524109, 203.2266846, -316.6167908, 310.1806641
2: -77.1763992, 176.2491455, -85.1790390, 194.8661194, -272.0425110, 261.4281311
3: -123.5897980, 213.3341370, -135.9188995, 235.9266510, -359.5164185, 349.2530518
4: -115.7205963, 203.1410065, -126.8144760, 224.1363678, -339.8569641, 329.9554749

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3196226, upper bound: 188.3217927
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3192050, upper bound: 188.3213658
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3220172, upper bound: 188.3241972
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -54.9176025, 143.7869873, -47.6437950, 124.7540054, -179.6715546, 191.4307709
1: -133.7262726, 214.3595123, -114.7317276, 186.0065308, -319.7327881, 329.0912476
2: -90.1855316, 206.5064087, -77.9661026, 178.1409149, -268.3264465, 284.4725037
3: -143.8749237, 249.4285583, -124.8423233, 215.4951782, -359.3700256, 374.2708740
4: -133.7892609, 237.2772827, -116.6950836, 205.1871796, -338.9764404, 353.9723511

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3206124, upper bound: 188.3186734
time: 1.07 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3200394, upper bound: 188.3159010
time: 1.14 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -55.5023804, 145.3056030, -48.7813225, 127.7284851, -183.2308350, 194.0869293
1: -135.1332092, 216.6195984, -117.3888397, 190.4262543, -325.5594177, 334.0084229
2: -91.1435242, 208.6767273, -79.8262329, 182.4131927, -273.5566406, 288.5029602
3: -145.4112701, 252.0180206, -127.8145218, 220.5801544, -365.9913635, 379.8324585
4: -135.2285614, 239.7668457, -119.5482025, 210.0509338, -345.2794800, 359.3149719

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3187801, upper bound: 188.3188691
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2607493, upper bound: 188.3161010
time: 1.49 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -54.5760117, 142.4660034, -50.4705658, 132.2124023, -186.7884064, 192.9365692
1: -132.6029663, 212.2594452, -121.3027344, 197.0949097, -329.6978760, 333.5621948
2: -89.5737762, 204.2021790, -82.5434875, 188.5274353, -278.1011658, 286.7456055
3: -142.9479370, 246.6837921, -132.2005768, 228.1289062, -371.0768433, 378.8843384
4: -133.0223846, 234.8143921, -123.7325058, 217.2451935, -350.2675781, 358.5468750

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2996545, upper bound: 188.2910730
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3137598, upper bound: 188.3186443
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -52.9937553, 138.1885986, -51.5986176, 134.5397644, -187.5334930, 189.7872009
1: -128.6731262, 205.9542999, -124.0797501, 200.6241913, -329.2972412, 330.0340576
2: -86.9595490, 198.0835571, -84.4321289, 191.8792725, -278.8388062, 282.5156860
3: -138.7703094, 239.2907104, -135.1500854, 232.1213837, -370.8916931, 374.4407959
4: -129.2381439, 227.7697906, -126.5433731, 221.0943146, -350.3324280, 354.3131714

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3027791, upper bound: 188.2918552
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3219645, upper bound: 188.3221393
time: 1.53 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -54.7518234, 142.8755035, -51.1240921, 133.8589325, -188.6107483, 193.9995880
1: -133.0482330, 212.8652954, -122.8309250, 199.5399933, -332.5882263, 335.6962280
2: -89.8336868, 204.9382782, -83.5895233, 190.8780823, -280.7117615, 288.5277100
3: -143.3931885, 247.5348969, -133.9144135, 230.9233704, -374.3165588, 381.4493103
4: -133.4600220, 235.5144806, -125.3740692, 219.9149017, -353.3749390, 360.8885193

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3235730, upper bound: 188.3232758
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3235730, upper bound: 188.3232758
time: 1.41 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -53.1935158, 138.6869354, -52.1151199, 135.8435211, -189.0370026, 190.8020325
1: -129.1500092, 206.7030640, -125.2480011, 202.5725708, -331.7225952, 331.9510498
2: -87.2518539, 198.9200592, -85.2480011, 193.7130737, -280.9649353, 284.1680603
3: -139.2792511, 240.2853546, -136.5116119, 234.3323517, -373.6116028, 376.7969360
4: -129.7458649, 228.6047058, -127.8711090, 223.1991119, -352.9449768, 356.4757690

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3247108, upper bound: 188.3233508
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3247108, upper bound: 188.3233508
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -50.9352722, 133.8157959, -44.6787796, 117.2538376, -168.1891022, 178.4945679
1: -123.4903946, 199.5901184, -107.3901672, 174.8407593, -298.3311157, 306.9802246
2: -83.4416809, 191.6258240, -73.0275726, 167.2903137, -250.7319794, 264.6533813
3: -133.0675201, 232.1209412, -116.9306717, 202.4448853, -335.5123901, 349.0515747
4: -123.8905182, 220.3979340, -109.3809662, 192.7294006, -316.6199341, 329.7789001

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3226652, upper bound: 188.3187772
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3226652, upper bound: 188.3187772
time: 0.90 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -51.5358772, 135.3626099, -45.8448601, 120.2485962, -171.7844543, 181.2074738
1: -124.9347763, 201.8900757, -110.1282120, 179.3171539, -304.2518921, 312.0182495
2: -84.4267883, 193.8346863, -74.9400177, 171.5880432, -256.0148315, 268.7746887
3: -134.6502686, 234.7458496, -119.9963074, 207.5867310, -342.2369995, 354.7421570
4: -125.3726120, 222.9287720, -112.2914047, 197.6067657, -322.9793396, 335.2201538

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3185244, upper bound: 188.3189298
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3185244, upper bound: 188.3189298
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -51.4432182, 135.0791779, -48.3047981, 127.2239685, -178.6671753, 183.3839569
1: -124.5319443, 201.4742889, -116.1450958, 189.8166199, -314.3485107, 317.6193542
2: -84.2193069, 193.2967682, -78.9903564, 181.4607697, -265.6800537, 272.2871094
3: -134.4287262, 234.0551758, -126.3980789, 219.7408142, -354.1694946, 360.4532471
4: -125.2743988, 222.3603058, -118.3185730, 209.1559601, -334.4303589, 340.6788940

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3205252, upper bound: 188.3200926
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3236252, upper bound: 188.3210281
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -51.7537842, 135.4740906, -46.1218719, 120.8167419, -172.5705261, 181.5959473
1: -125.1325302, 202.1875000, -110.7980118, 180.1492615, -305.2817993, 312.9855042
2: -84.7584457, 193.8899994, -75.3865356, 172.2470093, -257.0054321, 269.2764893
3: -135.2291412, 234.7633209, -120.6501999, 208.5083923, -343.7375488, 355.4134216
4: -126.1649017, 223.0133820, -112.9710312, 198.5586853, -324.7235718, 335.9844055

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3189616, upper bound: 188.3161754
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3202661, upper bound: 188.3097701
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3209771, upper bound: 188.3189161
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -52.3656731, 137.0442963, -47.2615509, 123.7452469, -176.1109161, 184.3058472
1: -126.5963974, 204.5211487, -113.5092621, 184.5224457, -311.1188354, 318.0303345
2: -85.7595215, 196.1338348, -77.2688065, 176.4466248, -262.2060852, 273.4026184
3: -136.8341980, 237.4304504, -123.6653595, 213.5435638, -350.3777466, 361.0958252
4: -127.6752090, 225.5767212, -115.8057404, 203.3301697, -331.0053711, 341.3823853

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3019575, upper bound: 188.2907962
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3232902, upper bound: 188.3209809
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -53.5206337, 139.7246399, -53.6546364, 140.1725311, -193.6931610, 193.3792572
1: -130.1225891, 208.1431274, -130.3436279, 208.8013000, -338.9238892, 338.4867554
2: -87.8495636, 200.3626556, -88.0557632, 200.7680054, -288.6175537, 288.4184265
3: -140.2376709, 241.9433136, -140.6176147, 242.4589081, -382.6965637, 382.5609131
4: -130.4658508, 230.3281708, -130.8421021, 230.9752808, -361.4411316, 361.1702881

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3203595, upper bound: 188.3114510
time: 1.27 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3203595, upper bound: 188.3258416
time: 1.33 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -55.0081406, 143.4553375, -67.2267609, 174.9165039, -229.9246063, 210.6820984
1: -133.8567352, 213.6765289, -163.6852264, 260.2478943, -394.1045837, 377.3617554
2: -90.3259201, 205.8610687, -110.4618149, 250.7398376, -341.0656738, 316.3228760
3: -144.1161652, 248.5207977, -176.4502869, 301.8004761, -445.9166260, 424.9710693
4: -134.0070190, 236.5381775, -163.6815491, 288.2557068, -422.2627258, 400.2196655

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3211367, upper bound: 188.3122109
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3256967, upper bound: 188.3258472
time: 1.06 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -54.4428787, 141.5781555, -53.1130562, 138.4369812, -192.8798370, 194.6912079
1: -132.3741760, 211.0117493, -129.2233582, 206.3311920, -338.7053833, 340.2351074
2: -89.4387283, 203.1370697, -87.2344742, 198.6028137, -288.0415344, 290.3714905
3: -142.6381989, 245.2841797, -139.1619415, 239.8242340, -382.4624329, 384.4461060
4: -132.7721558, 233.4505920, -129.4252319, 228.3144379, -361.0865784, 362.8758240

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3244637, upper bound: 188.3243966
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3244637, upper bound: 188.3243966
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -53.2756042, 138.6501465, -55.4676704, 144.5647583, -197.8403473, 194.1177979
1: -129.4237061, 206.7070160, -134.7464294, 215.3478088, -344.7714844, 341.4533386
2: -87.5105438, 198.9457703, -91.0920410, 207.3867493, -294.8972778, 290.0378113
3: -139.4714508, 240.2619171, -145.1672821, 250.2608643, -389.7322998, 385.4291992
4: -129.9032745, 228.6287842, -135.0962982, 238.2601013, -368.1633606, 363.7250671

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3252050, upper bound: 188.3257467
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3252050, upper bound: 188.3257467
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -54.2158508, 141.4879913, -53.9332008, 140.8036194, -195.0194702, 195.4211884
1: -131.7355804, 210.8255615, -131.1150513, 209.8165894, -341.5520325, 341.9406128
2: -88.9654007, 202.9573212, -88.5050735, 202.0254211, -290.9907532, 291.4623413
3: -142.0085754, 245.1465759, -141.2383881, 244.0280609, -386.0366211, 386.3849487
4: -132.1939392, 233.2334137, -131.4586639, 232.1340942, -364.3280334, 364.6920471

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3035831, upper bound: 188.3145671
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3133887, upper bound: 188.3222554
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -53.9253082, 140.7609253, -55.2297821, 144.6632843, -198.5885773, 195.9907074
1: -131.0251923, 209.7553406, -134.4703979, 215.6731720, -346.6983643, 344.2257385
2: -88.4889526, 201.9250336, -90.6771469, 207.7243652, -296.2132874, 292.6021729
3: -141.2384949, 243.9094543, -144.6972809, 250.8894806, -392.1279602, 388.6067200
4: -131.4831696, 232.0429688, -134.5488434, 238.7010498, -370.1842041, 366.5917969

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3261317, upper bound: 188.3259380
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3254942, upper bound: 188.3259301
time: 1.32 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -56.3240356, 147.1082153, -54.6144676, 142.5590057, -198.8830414, 201.7226868
1: -136.6592712, 219.2078400, -132.6358337, 212.4209595, -349.0802002, 351.8436279
2: -92.3859024, 210.7996826, -89.5863266, 204.4244690, -296.8103638, 300.3859863
3: -147.5200348, 254.7383118, -143.0403442, 246.9089355, -394.4289551, 397.7786255
4: -137.4270935, 242.3625488, -133.1888275, 234.9505768, -372.3775940, 375.5513916

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3265377, upper bound: 188.3274871
time: 1.01 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3262096, upper bound: 188.3272541
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -54.7457199, 142.8545074, -54.6166115, 142.1342163, -196.8799286, 197.4711151
1: -132.7383728, 212.9271240, -132.4591827, 211.8775635, -344.6158752, 345.3862915
2: -89.7765656, 204.6877594, -89.6067123, 203.7202148, -293.4967041, 294.2944641
3: -143.3602142, 247.3577881, -143.0356140, 246.0721741, -389.4323730, 390.3934021
4: -133.6569061, 235.3475952, -133.3600464, 234.1829834, -367.8398132, 368.7076416

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3265506, upper bound: 188.3272835
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3262654, upper bound: 188.3271782
time: 1.00 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -51.0959091, 134.1338348, -50.7181778, 133.2198334, -184.3157349, 184.8520050
1: -123.9878387, 200.0505676, -123.2150421, 198.7136841, -322.7015076, 323.2655945
2: -83.7308807, 192.0693970, -83.1649017, 191.0793915, -274.8102417, 275.2343140
3: -133.5570526, 232.5882568, -132.7419128, 231.0460815, -364.6031189, 365.3300781
4: -124.2374039, 220.9663086, -123.5410538, 219.7389221, -343.9763184, 344.5073547

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3195545, upper bound: 188.3190312
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3191201, upper bound: 188.3190312
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -53.4053459, 140.0942688, -48.7806664, 127.7638474, -181.1691895, 188.8749390
1: -129.1812134, 208.9654999, -118.1235275, 190.5844879, -319.7656555, 327.0889587
2: -87.4494019, 200.6733398, -79.9091949, 183.1181946, -270.5675964, 280.5824280
3: -139.3610992, 242.8500061, -127.4962845, 221.4170380, -360.7781372, 370.3462830
4: -129.8975220, 230.5849457, -118.9144592, 210.5176849, -340.4151917, 349.4993896

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2847439, upper bound: 188.2967060
time: 1.38 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2834647, upper bound: 188.2940771
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -53.0927010, 139.2917938, -49.8518066, 131.0369568, -184.1296539, 189.1435699
1: -128.4078674, 207.7785950, -120.9484634, 195.5622406, -323.9700928, 328.7269592
2: -86.9380951, 199.5379486, -81.7207718, 187.9947968, -274.9328308, 281.2586670
3: -138.5282898, 241.4772491, -130.3571625, 227.3046570, -365.8329468, 371.8344116
4: -129.1346893, 229.2629700, -121.4327545, 216.1372375, -345.2719116, 350.6957092

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3196374, upper bound: 188.3202766
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3190559, upper bound: 188.3190348
time: 1.29 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -50.7616386, 133.4487000, -50.7792435, 133.1243591, -183.8859711, 184.2279358
1: -122.8311081, 199.0320282, -122.7765656, 198.4444580, -321.2755127, 321.8085632
2: -83.1084824, 190.7433929, -83.1409683, 190.3635864, -273.4720764, 273.8843689
3: -132.6823578, 231.0183105, -132.8727264, 230.2639160, -362.9462891, 363.8910522
4: -123.6051712, 219.6548920, -123.9287796, 219.1335297, -342.7387085, 343.5836792

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3210183, upper bound: 188.3113332
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3239721, upper bound: 188.3229036
time: 1.23 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -52.1674271, 136.9917755, -63.5276794, 165.5996399, -217.7670593, 200.5194550
1: -126.4040298, 204.3027496, -154.3313599, 246.3877716, -372.7918091, 358.6340942
2: -85.4510651, 195.9908752, -104.2741394, 237.1699829, -322.6210327, 300.2650146
3: -136.3509521, 237.3208160, -166.5565033, 285.6577759, -422.0086975, 403.8773193
4: -126.9340897, 225.5775146, -154.6454163, 272.7597046, -399.6937256, 380.2229309

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3234139, upper bound: 188.3222078
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3240025, upper bound: 188.3239863
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -51.6724625, 135.6597290, -52.2704277, 136.8693848, -188.5418396, 187.9301605
1: -125.2618103, 202.3330078, -126.6716003, 204.0722046, -329.3340149, 329.0046082
2: -84.6323166, 194.2055664, -85.6592407, 196.0101929, -280.6423950, 279.8648071
3: -135.0440369, 235.1744995, -136.7565613, 237.0774384, -372.1214294, 371.9310303
4: -125.6867447, 223.4272461, -127.4256821, 225.4734039, -351.1601562, 350.8529358

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3264006, upper bound: 188.3265482
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3246530, upper bound: 188.3255383
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -50.8099327, 133.4263306, -54.3845139, 142.3276825, -193.1376190, 187.8108215
1: -122.9565659, 199.0967560, -131.5496521, 212.1672516, -335.1238098, 330.6464233
2: -83.1970825, 191.0232697, -89.0971069, 203.9153137, -287.1123962, 280.1203613
3: -132.6529541, 231.3013611, -142.0956879, 246.4195709, -379.0725098, 373.3970337
4: -123.6163025, 219.6592712, -132.5224762, 234.2968597, -357.9131470, 352.1817322

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3229519, upper bound: 188.3133236
time: 0.89 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3251360, upper bound: 188.3250725
time: 0.85 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.09 seconds
NS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3249859, upper bound: 188.3260581
NS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3251803, upper bound: 188.3239497
NS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3250934, upper bound: 188.3263103
NS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3250934, upper bound: 188.3240404
NS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3222588, upper bound: 188.3118520
NS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3222588, upper bound: 188.3222454
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3220140, upper bound: 188.3118520
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3263897, upper bound: 188.3222857
NS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3198992, upper bound: 188.3222713
NS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3201178, upper bound: 188.3191626
NS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3189187, upper bound: 188.3169385
NS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3191661, upper bound: 188.3168856
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3188654, upper bound: 188.3174735
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3188654, upper bound: 188.3174735
NS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3221660, upper bound: 188.3232758
NS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3221660, upper bound: 188.3232758
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3192671, upper bound: 188.3213168
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3214952, upper bound: 188.3225729
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3190405, upper bound: 188.3207923
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3211785, upper bound: 188.3215192
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3190887, upper bound: 188.3204521
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3212797, upper bound: 188.3212797
NS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3193381, upper bound: 188.3208008
NS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3193381, upper bound: 188.3212310
NS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3165451, upper bound: 188.3206378
NS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3165451, upper bound: 188.3208647
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3196529, upper bound: 188.3194879
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3196529, upper bound: 188.3196211
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.2708179, upper bound: 188.2820991
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.2589619, upper bound: 188.2596760
NS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3188222, upper bound: 188.3215458
NS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3188222, upper bound: 188.3215458
NS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3165451, upper bound: 188.3209351
NS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3165451, upper bound: 188.3213730
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3181953, upper bound: 188.3189197
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3190748, upper bound: 188.3188466
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3189050, upper bound: 188.3189197
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3190748, upper bound: 188.3188466
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.2801124, upper bound: 188.2689167
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3229697, upper bound: 188.3243913
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3194076, upper bound: 188.3204936
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3225729, upper bound: 188.3239720
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.2804533, upper bound: 188.2694271
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3218578, upper bound: 188.3243864
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3192050, upper bound: 188.3213658
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3220172, upper bound: 188.3241972
NS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3206124, upper bound: 188.3186734
NS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3200394, upper bound: 188.3159010
NS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3187801, upper bound: 188.3188691
NS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.2607493, upper bound: 188.3161010
NS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.2996545, upper bound: 188.2910730
NS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3137598, upper bound: 188.3186443
NS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3027791, upper bound: 188.2918552
NS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3219645, upper bound: 188.3221393
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3235730, upper bound: 188.3232758
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3235730, upper bound: 188.3232758
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3247108, upper bound: 188.3233508
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3247108, upper bound: 188.3233508
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3226652, upper bound: 188.3187772
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3226652, upper bound: 188.3187772
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3185244, upper bound: 188.3189298
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3185244, upper bound: 188.3189298
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3205252, upper bound: 188.3200926
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3236252, upper bound: 188.3210281
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3202661, upper bound: 188.3097701
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3209771, upper bound: 188.3189161
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3019575, upper bound: 188.2907962
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3232902, upper bound: 188.3209809
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3203595, upper bound: 188.3114510
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3203595, upper bound: 188.3258416
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3211367, upper bound: 188.3122109
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3256967, upper bound: 188.3258472
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3244637, upper bound: 188.3243966
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3244637, upper bound: 188.3243966
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3252050, upper bound: 188.3257467
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3252050, upper bound: 188.3257467
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3035831, upper bound: 188.3145671
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3133887, upper bound: 188.3222554
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3261317, upper bound: 188.3259380
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3254942, upper bound: 188.3259301
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3265377, upper bound: 188.3274871
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3262096, upper bound: 188.3272541
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3265506, upper bound: 188.3272835
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3262654, upper bound: 188.3271782
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3195545, upper bound: 188.3190312
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3191201, upper bound: 188.3190312
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.2847439, upper bound: 188.2967060
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.2834647, upper bound: 188.2940771
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3196374, upper bound: 188.3202766
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3190559, upper bound: 188.3190348
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3210183, upper bound: 188.3113332
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3239721, upper bound: 188.3229036
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3234139, upper bound: 188.3222078
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3240025, upper bound: 188.3239863
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3264006, upper bound: 188.3265482
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3246530, upper bound: 188.3255383
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3229519, upper bound: 188.3133236
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -188.3251360, upper bound: 188.3250725

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -41.5216789, 108.3324509, -42.1374207, 109.8145218, -151.3361969, 150.4698792
1: -100.1862717, 161.3318176, -101.6211166, 163.2819366, -263.4681702, 262.9529419
2: -68.0138245, 154.7909393, -68.9961243, 156.6082306, -224.6220398, 223.7870636
3: -108.8119888, 186.9798584, -110.5096130, 189.0349884, -297.8469543, 297.4894104
4: -101.5970688, 178.2143097, -103.1666946, 180.2709045, -281.8679504, 281.3810120

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3142612, upper bound: 188.3217683
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3232178, upper bound: 188.3243940
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -42.7044945, 111.3910980, -42.8514214, 111.6490173, -154.3535156, 154.2424774
1: -102.9299622, 165.8821869, -103.3219147, 166.0238953, -268.9538574, 269.2040710
2: -69.9400024, 159.1625671, -70.1551361, 159.2073364, -229.1473236, 229.3176727
3: -111.9268570, 192.1859589, -112.3898621, 192.1457672, -304.0726318, 304.5758057
4: -104.5670166, 183.1642303, -104.9310608, 183.2549896, -287.8220215, 288.0952759

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3251803, upper bound: 188.3233964
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3251803, upper bound: 188.3239497
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -42.1351624, 109.9427109, -44.5198898, 116.3318481, -158.4669647, 154.4626007
1: -101.6329803, 163.7592621, -107.2497253, 173.3272552, -274.9602051, 271.0089722
2: -68.9953461, 157.0927582, -72.8654022, 166.1155243, -235.1108704, 229.9581604
3: -110.4095917, 189.7520294, -116.6455078, 200.7475433, -311.1571350, 306.3975220
4: -103.1093216, 180.8852844, -109.0245667, 191.2582245, -294.3675537, 289.9098511

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3140916, upper bound: 188.3217687
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3233447, upper bound: 188.3245973
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -43.3276253, 113.0255280, -45.1746788, 118.0088959, -161.3364868, 158.2001495
1: -104.4061890, 168.3553619, -108.8235626, 175.8189087, -280.2250671, 277.1789246
2: -70.9440231, 161.5096283, -73.9353790, 168.5031738, -239.4472046, 235.4449921
3: -113.5481796, 195.0198364, -118.3735352, 203.5970764, -317.1452637, 313.3933716
4: -106.0943832, 185.8802032, -110.6319733, 193.9962158, -300.0906067, 296.5121765

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3143896, upper bound: 188.3210121
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3234304, upper bound: 188.3218954
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -47.4951172, 124.5345688, -44.0194321, 114.8181763, -162.3132935, 168.5539703
1: -114.2126083, 185.7949219, -106.1712570, 170.7694702, -284.9820557, 291.9661560
2: -77.6849823, 177.7440491, -72.0967560, 163.7378235, -241.4227905, 249.8407745
3: -124.4524078, 215.0525513, -115.4489517, 197.6933289, -322.1457214, 330.5014954
4: -116.4170685, 204.7463226, -107.7210922, 188.4957275, -304.9127808, 312.4673767

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3212091, upper bound: 188.3102004
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3213362, upper bound: 188.3108897
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -48.4992790, 127.6415405, -43.7849770, 114.2145691, -162.7138519, 171.4265137
1: -116.7486496, 190.5939178, -105.5945206, 169.8818970, -286.6305237, 296.1883850
2: -79.3594742, 182.3659210, -71.7112045, 162.8950806, -242.2545471, 254.0770874
3: -127.0973053, 220.6434479, -114.8217468, 196.6772766, -323.7745361, 335.4652100
4: -118.8456650, 210.0911713, -107.1374054, 187.5093231, -306.3549805, 317.2285767

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3257329, upper bound: 188.3210926
time: 1.37 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3236117, upper bound: 188.3213274
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -48.1518974, 126.2609482, -47.0993805, 123.5255127, -171.6773834, 173.3603210
1: -115.7767715, 188.3895111, -113.3373947, 184.1966553, -299.9734192, 301.7268982
2: -78.7475662, 180.2496948, -77.0471878, 176.2910461, -255.0386047, 257.2968140
3: -126.1683502, 218.0399628, -123.4303970, 213.2680359, -339.4364014, 341.4703369
4: -118.0189209, 207.6173706, -115.4261932, 203.0646362, -321.0835571, 323.0434875

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3170920, upper bound: 188.3114058
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3170920, upper bound: 188.3118520
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -49.2554626, 129.6101990, -46.8782692, 122.9608078, -172.2162476, 176.4884644
1: -118.6022873, 193.5312347, -112.8008041, 183.3589020, -301.9611816, 306.3320312
2: -80.6002274, 185.2532501, -76.6874771, 175.5013885, -256.1016235, 261.9407349
3: -129.0781403, 224.0772552, -122.8274002, 212.3119049, -341.3900452, 346.9046631
4: -120.6557999, 213.3966980, -114.8747711, 202.1430969, -322.7988892, 328.2714844

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3256017, upper bound: 188.3210949
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3236111, upper bound: 188.3213719
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -46.7279358, 122.3938751, -49.3220787, 129.6116791, -176.3396149, 171.7159424
1: -112.5294342, 182.5096130, -118.9573288, 193.3623199, -305.8917542, 301.4669495
2: -76.4594421, 174.7452393, -80.7358475, 185.2304230, -261.6898804, 255.4810791
3: -122.4371109, 211.4839172, -129.2568970, 224.0324860, -346.4696045, 340.7408142
4: -114.4279633, 201.3233795, -120.7308807, 213.3627319, -327.7906799, 322.0542603

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3198992, upper bound: 188.3222714
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3198992, upper bound: 188.3222714
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -47.8586922, 125.3537292, -49.9341927, 131.1974487, -179.0560913, 175.2879181
1: -115.1776276, 186.9090729, -120.4302597, 195.7147217, -310.8923035, 307.3392944
2: -78.3099213, 179.0015869, -81.7367249, 187.4897461, -265.7996521, 260.7383118
3: -125.3938828, 216.5439606, -130.8665466, 226.7243500, -352.1182251, 347.4105225
4: -117.2621460, 206.1701660, -122.2358627, 215.9596558, -333.2217712, 328.4059448

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3201178, upper bound: 188.3191626
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3201178, upper bound: 188.3191626
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -46.5726776, 122.6101379, -45.7250137, 119.8897858, -166.4624634, 168.3351288
1: -111.8348312, 182.9411621, -110.0829239, 178.7098999, -290.5447388, 293.0240784
2: -76.0859451, 174.6318054, -74.7654800, 171.0998993, -247.1858368, 249.3972626
3: -121.8249130, 211.7287903, -119.6890869, 206.9720154, -328.7969360, 331.4178467
4: -113.8987579, 201.2890778, -111.8201218, 197.1266479, -311.0253906, 313.1091919

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3189187, upper bound: 188.3169385
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3189187, upper bound: 188.3169385
time: 1.10 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -46.8492813, 123.4809036, -45.0606232, 118.2259140, -165.0751648, 168.5415192
1: -112.5706940, 184.4990997, -108.4863434, 176.3003845, -288.8710327, 292.9854431
2: -76.5721741, 176.1124115, -73.6674194, 168.7617798, -245.3339539, 249.7798157
3: -122.5848999, 213.6325378, -117.9506149, 204.1794128, -326.7643127, 331.5831604
4: -114.5996552, 202.9921722, -110.2069092, 194.4431000, -309.0427551, 313.1990662

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3191661, upper bound: 188.3168856
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3189187, upper bound: 188.3168856
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -45.8030586, 120.6436844, -47.0310020, 123.2438965, -169.0469513, 167.6746826
1: -110.0646057, 180.0055695, -113.1517105, 183.7160950, -293.7807007, 293.1572876
2: -74.8493576, 171.8573456, -76.9089661, 175.9094391, -250.7587891, 248.7663116
3: -119.7982330, 208.4128265, -123.1244354, 212.7243652, -332.5225830, 331.5371399
4: -111.9513016, 198.1424561, -115.0782242, 202.5919037, -314.5430908, 313.2206726

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3170786, upper bound: 188.3174313
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3170786, upper bound: 188.3174313
time: 1.00 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -47.0601845, 123.8620605, -47.0310020, 123.2438965, -170.3040771, 170.8930206
1: -113.0181808, 184.8182983, -113.1517105, 183.7160950, -296.7342834, 297.9700012
2: -76.9098358, 176.4811249, -76.9089661, 175.9094391, -252.8192596, 253.3900909
3: -123.0926666, 213.9305267, -123.1244354, 212.7243652, -335.8170166, 337.0548706
4: -115.0977173, 203.3919220, -115.0782242, 202.5919037, -317.6896057, 318.4701538

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3170786, upper bound: 188.3170786
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3170786, upper bound: 188.3170786
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -50.5617332, 132.3796387, -49.4436378, 129.3868866, -179.9485931, 181.8232574
1: -121.4641190, 197.3448486, -118.9387665, 192.8413086, -314.3054199, 316.2836304
2: -82.6709290, 188.7581635, -80.8467560, 184.6259766, -267.2969055, 269.6049194
3: -132.4386597, 228.4206390, -129.5076141, 223.2431030, -355.6817627, 357.9282532
4: -123.9856873, 217.5033112, -121.1414185, 212.6535950, -336.6392822, 338.6447144

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3206951, upper bound: 188.3209081
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3213400, upper bound: 188.3224614
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -51.6164513, 134.5352173, -49.4436378, 129.3868866, -181.0033112, 183.9788513
1: -124.0399857, 200.6360168, -118.9387665, 192.8413086, -316.8812866, 319.5747681
2: -84.4323807, 191.8159027, -80.8467560, 184.6259766, -269.0583496, 272.6625977
3: -135.2171021, 232.0989685, -129.5076141, 223.2431030, -358.4602051, 361.6065674
4: -126.6499634, 221.0501709, -121.1414185, 212.6535950, -339.3035278, 342.1915894

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3206951, upper bound: 188.3209081
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3213400, upper bound: 188.3224614
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -48.2392044, 126.1158905, -49.3975143, 128.7437286, -176.9829407, 175.5133972
1: -115.9551926, 188.0398102, -118.7490845, 191.9963531, -307.9515381, 306.7888794
2: -78.8639297, 179.8392944, -80.7897186, 183.6532135, -262.5171204, 260.6290283
3: -126.2859573, 217.6873322, -129.3633728, 222.1656952, -348.4516602, 347.0506592
4: -118.2440262, 207.3009338, -121.1223221, 211.5898590, -329.8338318, 328.4232483

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3184351, upper bound: 188.3191616
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3184351, upper bound: 188.3213168
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -49.1721230, 128.5721283, -50.0280457, 130.3871613, -179.5592804, 178.6001740
1: -118.1451416, 191.7068329, -120.2678833, 194.4436646, -312.5888062, 311.9747009
2: -80.3986130, 183.3963165, -81.8195496, 186.0084381, -266.4069519, 265.2158203
3: -128.7115936, 221.9329529, -131.0242767, 224.9708099, -353.6824036, 352.9572144
4: -120.5682373, 211.3365326, -122.6719131, 214.2832489, -334.8515015, 334.0084229

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3204521, upper bound: 188.3194076
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3204521, upper bound: 188.3225729
time: 1.07 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -47.6065025, 125.4730072, -45.8666458, 120.2417831, -167.8482666, 171.3396606
1: -114.3563614, 187.1812286, -110.1986618, 179.2391205, -293.5954895, 297.3798828
2: -77.8104858, 178.6148224, -74.9467773, 171.4443207, -249.2547913, 253.5615997
3: -124.4757996, 216.7230835, -120.0277786, 207.4207306, -331.8965149, 336.7508240
4: -116.4402008, 206.0356293, -112.3100128, 197.5326233, -313.9728088, 318.3456421

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3188152, upper bound: 188.3198628
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3188152, upper bound: 188.3207923
time: 1.10 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -48.9023209, 128.7886200, -46.5276375, 121.9305191, -170.8328094, 175.3162537
1: -117.4100571, 192.1429138, -111.7981567, 181.7564697, -299.1664734, 303.9410706
2: -79.9399490, 183.3730164, -76.0321808, 173.8580780, -253.7980347, 259.4051514
3: -127.8765106, 222.4163818, -121.7752609, 210.3092499, -338.1857605, 344.1916504
4: -119.6803360, 211.4475708, -113.9303436, 200.3000641, -319.9804077, 325.3778992

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2815154, upper bound: 188.2935466
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3202702, upper bound: 188.3206319
time: 1.12 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -46.4082108, 122.2008362, -46.6140099, 121.7149658, -168.1231689, 168.8148041
1: -111.4122620, 182.3895111, -111.9611893, 181.5595551, -292.9718018, 294.3506470
2: -75.8368607, 173.9581146, -76.1953812, 173.5600281, -249.3968811, 250.1534882
3: -121.3025284, 211.0911407, -121.9897842, 210.0410156, -331.3435059, 333.0809326
4: -113.5616608, 200.6849060, -114.1970596, 199.9736481, -313.5352783, 314.8819275

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3183099, upper bound: 188.3183099
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3183099, upper bound: 188.3204521
time: 0.81 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.96 + 418.26 = 421.22 seconds
