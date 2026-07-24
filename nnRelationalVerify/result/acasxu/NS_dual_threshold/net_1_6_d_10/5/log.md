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
execution time: IAR + RelationalAnalysis = 0.73 + 2.15 = 2.87 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -188.3342834, upper bound: 188.3342834

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

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
- Time for NS candidates: 2.07 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.07
Output dim: 0, lower bound: -188.3300130, upper bound: 188.3328293
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.07
Output dim: 0, lower bound: -188.3342834, upper bound: 188.3342834

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -53.3530693, 139.3271179, -56.8082809, 148.0125580, -201.3656006, 196.1354065
1: -128.6982117, 207.5555573, -138.2005615, 220.4711914, -349.1694031, 345.7560120
2: -87.3666611, 199.1090698, -93.3056946, 212.5586548, -299.9253235, 292.4147644
3: -139.7742920, 240.3986511, -148.8231964, 256.5381470, -396.3124390, 389.2218323
4: -130.5633087, 229.1490479, -138.4369507, 244.0570221, -374.6203308, 367.5859985

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

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
time: 0.73 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -58.5414886, 152.5739136, -58.9514084, 153.6189880, -212.1604767, 211.5253143
1: -142.4417877, 227.2532349, -143.4570923, 228.8044281, -371.2462158, 370.7103271
2: -96.1404495, 219.0275879, -96.8237000, 220.5256195, -316.6660767, 315.8512878
3: -153.3723450, 264.2421570, -154.4591522, 266.0426025, -419.4149475, 418.7012024
4: -142.6441956, 251.5583649, -143.6370239, 253.2858734, -395.9300232, 395.1953735

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

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
- Time for NS candidates: 2.35 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -188.3299582, upper bound: 188.3299582
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -188.3299582, upper bound: 188.3328293
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -188.3328293, upper bound: 188.3300130
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -188.3328293, upper bound: 188.3342834

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -53.3530693, 139.3271179, -53.3530693, 139.3271179, -192.6801758, 192.6801758
1: -128.6982117, 207.5555573, -128.6982117, 207.5555573, -336.2537842, 336.2537842
2: -87.3666611, 199.1090698, -87.3666611, 199.1090698, -286.4757385, 286.4757385
3: -139.7742920, 240.3986511, -139.7742920, 240.3986511, -380.1729431, 380.1729431
4: -130.5633087, 229.1490479, -130.5633087, 229.1490479, -359.7123413, 359.7123413

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3252270, upper bound: 188.3210066
time: 1.13 seconds

## Relational analysis of NS_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3049762, upper bound: 188.3109311
time: 1.27 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -53.3530693, 139.3271179, -58.5414886, 152.5739136, -205.9269409, 197.8686066
1: -128.6982117, 207.5555573, -142.4417877, 227.2532349, -355.9514465, 349.9973450
2: -87.3666611, 199.1090698, -96.1404495, 219.0275879, -306.3942566, 295.2495117
3: -139.7742920, 240.3986511, -153.3723450, 264.2421570, -404.0163879, 393.7709961
4: -130.5633087, 229.1490479, -142.6441956, 251.5583649, -382.1216736, 371.7932434

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3075951, upper bound: 188.3162752
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3049762, upper bound: 188.3109311
time: 0.68 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -58.5414886, 152.5739136, -53.3530693, 139.3271179, -197.8686066, 205.9269409
1: -142.4417877, 227.2532349, -128.6982117, 207.5555573, -349.9973450, 355.9514465
2: -96.1404495, 219.0275879, -87.3666611, 199.1090698, -295.2495117, 306.3942566
3: -153.3723450, 264.2421570, -139.7742920, 240.3986511, -393.7709961, 404.0164185
4: -142.6441956, 251.5583649, -130.5633087, 229.1490479, -371.7932434, 382.1216736

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3162752, upper bound: 188.3075951
time: 1.02 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3042660, upper bound: 188.3042660
time: 0.78 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -58.5414886, 152.5739136, -58.5414886, 152.5739136, -211.1153870, 211.1153870
1: -142.4417877, 227.2532349, -142.4417877, 227.2532349, -369.6950073, 369.6950073
2: -96.1404495, 219.0275879, -96.1404495, 219.0275879, -315.1679993, 315.1679993
3: -153.3723450, 264.2421570, -153.3723450, 264.2421570, -417.6145020, 417.6145020
4: -142.6441956, 251.5583649, -142.6441956, 251.5583649, -394.2025452, 394.2025452

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3162752, upper bound: 188.3075951
time: 0.82 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3042660, upper bound: 188.3042660
time: 1.62 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.43 seconds
NS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -188.3252270, upper bound: 188.3210066
NS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -188.3049762, upper bound: 188.3109311
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -188.3075951, upper bound: 188.3162752
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -188.3049762, upper bound: 188.3109311
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -188.3162752, upper bound: 188.3075951
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -188.3042660, upper bound: 188.3042660
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -188.3162752, upper bound: 188.3075951
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -188.3042660, upper bound: 188.3042660

## BFS NS instance: NS_A1_B1_B1

### Backsubstitution after applying NS history:
0: -53.3530693, 139.3271179, -53.2309341, 139.0255432, -192.3785706, 192.5580444
1: -128.6982117, 207.5555573, -128.3856354, 207.1139984, -335.8121948, 335.9411316
2: -87.3666611, 199.1090698, -87.1618500, 198.6638794, -286.0305481, 286.2709351
3: -139.7742920, 240.3986511, -139.4619446, 239.8694763, -379.6437683, 379.8605957
4: -130.5633087, 229.1490479, -130.2797546, 228.6415558, -359.2048645, 359.4288025

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3253405, upper bound: 188.3230312
time: 0.78 seconds

## Relational analysis of NS_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3253406, upper bound: 188.3225489
time: 1.16 seconds

## BFS NS instance: NS_A1_B1_B2

### Backsubstitution after applying NS history:
0: -53.1794052, 138.8796692, -58.4173241, 152.8462524, -206.0256348, 197.2969818
1: -128.2687988, 206.8917236, -139.6684265, 227.0411072, -355.3098755, 346.5601501
2: -87.0837784, 198.4602814, -95.2001114, 217.2766876, -304.3604431, 293.6603394
3: -139.3088837, 239.6149139, -152.8114166, 262.6232910, -401.9321899, 392.4263306
4: -130.1359711, 228.4168854, -143.5048218, 250.1871185, -380.3230896, 371.9216919

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3217783, upper bound: 188.3217783
time: 0.76 seconds

## Relational analysis of NS_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3217783, upper bound: 188.3217783
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -53.2309341, 139.0255432, -58.5414886, 152.5739136, -205.8048248, 197.5670166
1: -128.3856354, 207.1139984, -142.4417877, 227.2532349, -355.6388550, 349.5557861
2: -87.1618500, 198.6638794, -96.1404495, 219.0275879, -306.1894226, 294.8043213
3: -139.4619446, 239.8694763, -153.3723450, 264.2421570, -403.7040405, 393.2418213
4: -130.2797546, 228.6415558, -142.6441956, 251.5583649, -381.8381348, 371.2857056

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2846687, upper bound: 188.2946892
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2769715, upper bound: 188.2886197
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -58.4173241, 152.8462524, -58.3344421, 152.0396729, -210.4569855, 211.1806946
1: -139.6684265, 227.0411072, -141.9337158, 226.4614258, -366.1298523, 368.9747620
2: -95.2001114, 217.2766876, -95.8018188, 218.2637787, -313.4638977, 313.0784912
3: -152.8114166, 262.6232910, -152.8282471, 263.3311462, -416.1425781, 415.4515381
4: -143.5048218, 250.1871185, -142.1469269, 250.6787872, -394.1835022, 392.3340454

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3049762, upper bound: 188.3109311
time: 1.20 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3049762, upper bound: 188.3109311
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -58.5414886, 152.5739136, -53.2309341, 139.0255432, -197.5670166, 205.8048248
1: -142.4417877, 227.2532349, -128.3856354, 207.1139984, -349.5557861, 355.6388550
2: -96.1404495, 219.0275879, -87.1618500, 198.6638794, -294.8043213, 306.1894531
3: -153.3723450, 264.2421570, -139.4619446, 239.8694763, -393.2418213, 403.7040100
4: -142.6441956, 251.5583649, -130.2797546, 228.6415558, -371.2857056, 381.8381348

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2946892, upper bound: 188.2846687
time: 0.97 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2886197, upper bound: 188.2769715
time: 1.01 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -58.3344421, 152.0396729, -58.4173241, 152.8462524, -211.1806946, 210.4569855
1: -141.9337158, 226.4614258, -139.6684265, 227.0411072, -368.9747925, 366.1298523
2: -95.8018188, 218.2637787, -95.2001114, 217.2766876, -313.0784912, 313.4638977
3: -152.8282471, 263.3311462, -152.8114166, 262.6232910, -415.4515381, 416.1425781
4: -142.1469269, 250.6787872, -143.5048218, 250.1871185, -392.3340454, 394.1835022

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3109311, upper bound: 188.3049762
time: 0.73 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3109311, upper bound: 188.3049762
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: -58.5414886, 152.5739136, -58.4161682, 152.2642517, -210.8057404, 210.9900513
1: -142.4417877, 227.2532349, -142.1168060, 226.7994690, -369.2412720, 369.3700256
2: -96.1404495, 219.0275879, -95.9289169, 218.5611115, -314.7015686, 314.9564514
3: -153.3723450, 264.2421570, -153.0491333, 263.6882324, -417.0605774, 417.2912903
4: -142.6441956, 251.5583649, -142.3532867, 251.0381012, -393.6823120, 393.9116516

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_B1_B1

### Relational analysis result of NS_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3025031, upper bound: 188.2902226
time: 1.82 seconds

## Relational analysis of NS_A2_B2_B1_B2

### Relational analysis result of NS_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3162752, upper bound: 188.3075951
time: 1.28 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: -58.3344421, 152.0396729, -63.1659355, 164.9387207, -223.2731628, 215.2055969
1: -141.9337158, 226.4614258, -152.4431305, 245.0577850, -386.9914246, 378.9045410
2: -95.8018188, 218.2637787, -103.2844696, 235.5187683, -331.3205872, 321.5482483
3: -152.8282471, 263.3311462, -165.2579346, 284.5217896, -437.3500061, 428.5890808
4: -142.1469269, 250.6787872, -154.5637207, 270.7111511, -412.8580627, 405.2424316

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3042660, upper bound: 188.3042660
time: 0.83 seconds

## Relational analysis of NS_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3042660, upper bound: 188.3042660
time: 0.78 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.39 seconds
NS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -188.3253405, upper bound: 188.3230312
NS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -188.3253406, upper bound: 188.3225489
NS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -188.3217783, upper bound: 188.3217783
NS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -188.3217783, upper bound: 188.3217783
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.39
Output dim: 0, lower bound: -188.2846687, upper bound: 188.2946892
NS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.39
Output dim: 0, lower bound: -188.2769715, upper bound: 188.2886197
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -188.3049762, upper bound: 188.3109311
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -188.3049762, upper bound: 188.3109311
NS_A2_B1_B1_A1, status: Status.VERIFIED, split count: 4, time: 2.39
Output dim: 0, lower bound: -188.2946892, upper bound: 188.2846687
NS_A2_B1_B1_A2, status: Status.VERIFIED, split count: 4, time: 2.39
Output dim: 0, lower bound: -188.2886197, upper bound: 188.2769715
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -188.3109311, upper bound: 188.3049762
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -188.3109311, upper bound: 188.3049762
NS_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -188.3025031, upper bound: 188.2902226
NS_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -188.3162752, upper bound: 188.3075951
NS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -188.3042660, upper bound: 188.3042660
NS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -188.3042660, upper bound: 188.3042660

## BFS NS instance: NS_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -51.1813354, 133.8749084, -51.9747353, 135.8705139, -187.0518494, 185.8496246
1: -123.1871643, 199.5406036, -125.1993103, 202.4816437, -325.6687317, 324.7398376
2: -83.7470856, 191.1570892, -85.0678711, 194.0596771, -277.8067627, 276.2249451
3: -134.1554108, 230.9798126, -136.2134247, 234.4198456, -368.5751953, 367.1932068
4: -125.4367523, 220.0620270, -127.3158188, 223.3735199, -348.8102722, 347.3778381

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_B1_A1_B1

### Relational analysis result of NS_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3253406, upper bound: 188.3225104
time: 0.70 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2

### Relational analysis result of NS_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3253406, upper bound: 188.3225489
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -53.2817039, 139.3572388, -51.9709511, 135.8224030, -189.1041107, 191.3281708
1: -128.1620789, 207.6663666, -125.1434021, 202.3800201, -330.5420532, 332.8097534
2: -87.1664886, 198.8413086, -85.0368729, 193.9407501, -281.1072083, 283.8781738
3: -139.5358276, 240.3794708, -136.1020203, 234.3127899, -373.8486328, 376.4814758
4: -130.5832977, 229.0112457, -127.2784882, 223.2863312, -353.8695679, 356.2897034

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_B1_A2_B1

### Relational analysis result of NS_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3253405, upper bound: 188.3225104
time: 0.69 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2

### Relational analysis result of NS_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3253406, upper bound: 188.3225489
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -53.2309341, 139.0255432, -58.4173241, 152.8462524, -206.0771790, 197.4428406
1: -128.3856354, 207.1139984, -139.6684265, 227.0411072, -355.4267273, 346.7824097
2: -87.1618500, 198.6638794, -95.2001114, 217.2766876, -304.4385376, 293.8639832
3: -139.4619446, 239.8694763, -152.8114166, 262.6232910, -402.0852356, 392.6809082
4: -130.2797546, 228.6415558, -143.5048218, 250.1871185, -380.4668579, 372.1463318

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_B2_A1_A1

### Relational analysis result of NS_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3197422, upper bound: 188.3215807
time: 0.80 seconds

## Relational analysis of NS_A1_B1_B2_A1_A2

### Relational analysis result of NS_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3205367, upper bound: 188.3205367
time: 1.15 seconds

## BFS NS instance: NS_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -58.4173241, 152.8462524, -58.4173241, 152.8462524, -211.2635651, 211.2635651
1: -139.6684265, 227.0411072, -139.6684265, 227.0411072, -366.7095337, 366.7095337
2: -95.2001114, 217.2766876, -95.2001114, 217.2766876, -312.4767761, 312.4768066
3: -152.8114166, 262.6232910, -152.8114166, 262.6232910, -415.4346924, 415.4346924
4: -143.5048218, 250.1871185, -143.5048218, 250.1871185, -393.6918945, 393.6918945

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_B2_A2_A1

### Relational analysis result of NS_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3197422, upper bound: 188.3215807
time: 0.73 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2

### Relational analysis result of NS_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3205367, upper bound: 188.3205367
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -58.4173241, 152.8462524, -58.4161682, 152.2642517, -210.6815643, 211.2624054
1: -139.6684265, 227.0411072, -142.1168060, 226.7994690, -366.4678955, 369.1578674
2: -95.2001114, 217.2766876, -95.9289169, 218.5611115, -313.7612305, 313.2055359
3: -152.8114166, 262.6232910, -153.0491333, 263.6882324, -416.4996338, 415.6724243
4: -143.5048218, 250.1871185, -142.3532867, 251.0381012, -394.5429077, 392.5404053

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2705781, upper bound: 188.2795624
time: 1.08 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2705675, upper bound: 188.2795005
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -58.4173241, 152.8462524, -63.1659355, 164.9387207, -223.3560333, 216.0121613
1: -139.6684265, 227.0411072, -152.4431305, 245.0577850, -384.7261963, 379.4842529
2: -95.2001114, 217.2766876, -103.2844696, 235.5187683, -330.7188721, 320.5611572
3: -152.8114166, 262.6232910, -165.2579346, 284.5217896, -437.3331909, 427.8812256
4: -143.5048218, 250.1871185, -154.5637207, 270.7111511, -414.2159119, 404.7508240

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B2_B1

### Relational analysis result of NS_A1_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2756901, upper bound: 188.2770984
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2

### Relational analysis result of NS_A1_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2726798, upper bound: 188.2772289
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -58.4161682, 152.2642517, -58.4173241, 152.8462524, -211.2624054, 210.6815643
1: -142.1168060, 226.7994690, -139.6684265, 227.0411072, -369.1578674, 366.4678955
2: -95.9289169, 218.5611115, -95.2001114, 217.2766876, -313.2055969, 313.7612305
3: -153.0491333, 263.6882324, -152.8114166, 262.6232910, -415.6724243, 416.4996338
4: -142.3532867, 251.0381012, -143.5048218, 250.1871185, -392.5404053, 394.5429077

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2795624, upper bound: 188.2705781
time: 0.70 seconds

## Relational analysis of NS_A2_B1_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2795005, upper bound: 188.2705675
time: 1.37 seconds

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -63.1659355, 164.9387207, -58.4173241, 152.8462524, -216.0121613, 223.3560486
1: -152.4431305, 245.0577850, -139.6684265, 227.0411072, -379.4842529, 384.7261963
2: -103.2844696, 235.5187683, -95.2001114, 217.2766876, -320.5611572, 330.7188721
3: -165.2579346, 284.5217896, -152.8114166, 262.6232910, -427.8812256, 437.3331909
4: -154.5637207, 270.7111511, -143.5048218, 250.1871185, -404.7508545, 414.2159119

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_B2_A2_A1

### Relational analysis result of NS_A2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2770984, upper bound: 188.2756901
time: 0.72 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2

### Relational analysis result of NS_A2_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2772289, upper bound: 188.2726798
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -56.6906242, 147.8851318, -55.5342064, 144.9453430, -201.6359711, 203.4193420
1: -137.8075104, 220.2702484, -134.8983002, 215.8840332, -353.6915283, 355.1685486
2: -93.0620804, 212.1294861, -91.1362610, 207.7673187, -300.8294067, 303.2657166
3: -148.5431519, 255.9613495, -145.5397491, 250.7280121, -399.2710876, 401.5010986
4: -138.2164917, 243.7522583, -135.4631958, 238.8403931, -377.0568848, 379.2154541

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_B1_B1_A1

### Relational analysis result of NS_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3020126, upper bound: 188.2897183
time: 1.16 seconds

## Relational analysis of NS_A2_B2_B1_B1_A2

### Relational analysis result of NS_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3020126, upper bound: 188.2902226
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -58.2001610, 151.6906128, -68.3880310, 177.8731689, -236.0733337, 220.0786438
1: -141.5930023, 225.9341278, -166.5200500, 264.5990906, -406.1920776, 392.4541321
2: -95.5727463, 217.7410126, -112.3821106, 255.0892029, -350.6619568, 330.1231079
3: -152.4815063, 262.6842041, -179.4766388, 306.8139648, -459.2954712, 442.1608276
4: -141.8224792, 250.0854492, -166.5419922, 293.1131897, -434.9356689, 416.6274414

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_B1_B2_A1

### Relational analysis result of NS_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3021248, upper bound: 188.2899172
time: 0.73 seconds

## Relational analysis of NS_A2_B2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_B1_B2_B1

### Relational analysis result of NS_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3118825, upper bound: 188.3028176
time: 1.04 seconds

## Relational analysis of NS_A2_B2_B1_B2_B2

### Relational analysis result of NS_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3115452, upper bound: 188.3016574
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -58.4161682, 152.2642517, -63.1659355, 164.9387207, -223.3548889, 215.4301605
1: -142.1168060, 226.7994690, -152.4431305, 245.0577850, -387.1745605, 379.2426147
2: -95.9289169, 218.5611115, -103.2844696, 235.5187683, -331.4476929, 321.8455811
3: -153.0491333, 263.6882324, -165.2579346, 284.5217896, -437.5709229, 428.9461670
4: -142.3532867, 251.0381012, -154.5637207, 270.7111511, -413.0644226, 405.6018066

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_B2_A1_B1

### Relational analysis result of NS_A2_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2760395, upper bound: 188.2760395
time: 0.90 seconds

## Relational analysis of NS_A2_B2_B2_A1_B2

### Relational analysis result of NS_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3041004, upper bound: 188.3041004
time: 1.10 seconds

## BFS NS instance: NS_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -63.1659355, 164.9387207, -63.1659355, 164.9387207, -228.1046295, 228.1046448
1: -152.4431305, 245.0577850, -152.4431305, 245.0577850, -397.5009155, 397.5009155
2: -103.2844696, 235.5187683, -103.2844696, 235.5187683, -338.8032227, 338.8032227
3: -165.2579346, 284.5217896, -165.2579346, 284.5217896, -449.7797241, 449.7797241
4: -154.5637207, 270.7111511, -154.5637207, 270.7111511, -425.2748413, 425.2748718

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_B2_A2_A1

### Relational analysis result of NS_A2_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2951218, upper bound: 188.2871117
time: 0.84 seconds

## Relational analysis of NS_A2_B2_B2_A2_A2

### Relational analysis result of NS_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3042660, upper bound: 188.3042660
time: 1.03 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.67 seconds
NS_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -188.3253406, upper bound: 188.3225104
NS_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -188.3253406, upper bound: 188.3225489
NS_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -188.3253405, upper bound: 188.3225104
NS_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -188.3253406, upper bound: 188.3225489
NS_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -188.3197422, upper bound: 188.3215807
NS_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -188.3205367, upper bound: 188.3205367
NS_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -188.3197422, upper bound: 188.3215807
NS_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -188.3205367, upper bound: 188.3205367
NS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -188.2705781, upper bound: 188.2795624
NS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -188.2705675, upper bound: 188.2795005
NS_A1_B2_A2_B2_B1, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -188.2756901, upper bound: 188.2770984
NS_A1_B2_A2_B2_B2, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -188.2726798, upper bound: 188.2772289
NS_A2_B1_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -188.2795624, upper bound: 188.2705781
NS_A2_B1_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -188.2795005, upper bound: 188.2705675
NS_A2_B1_B2_A2_A1, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -188.2770984, upper bound: 188.2756901
NS_A2_B1_B2_A2_A2, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -188.2772289, upper bound: 188.2726798
NS_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -188.3020126, upper bound: 188.2897183
NS_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -188.3020126, upper bound: 188.2902226
NS_A2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -188.3118825, upper bound: 188.3028176
NS_A2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -188.3115452, upper bound: 188.3016574
NS_A2_B2_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -188.2760395, upper bound: 188.2760395
NS_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -188.3041004, upper bound: 188.3041004
NS_A2_B2_B2_A2_A1, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -188.2951218, upper bound: 188.2871117
NS_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -188.3042660, upper bound: 188.3042660

## BFS NS instance: NS_A1_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -51.1813354, 133.8749084, -51.0635910, 133.5844116, -184.7657471, 184.9384918
1: -123.1871643, 199.5406036, -122.8893051, 199.1136322, -322.3007812, 322.4298706
2: -83.7470856, 191.1570892, -83.5510254, 190.7286835, -274.4756775, 274.7080994
3: -134.1554108, 230.9798126, -133.8546753, 230.4699707, -364.6253357, 364.8344727
4: -125.4367523, 220.0620270, -125.1628265, 219.5740967, -345.0108643, 345.2247925

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3206434, upper bound: 188.3204136
time: 0.75 seconds

## Relational analysis of NS_A1_B1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3206434, upper bound: 188.3207773
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -51.1813354, 133.8749084, -53.1687355, 139.0787506, -190.2600861, 187.0436401
1: -123.1871643, 199.5406036, -127.8708878, 207.2600708, -330.4472351, 327.4114380
2: -83.7470856, 191.1570892, -86.9767456, 198.4295807, -282.1765747, 278.1338501
3: -134.1554108, 230.9798126, -139.2464447, 239.8921814, -374.0475159, 370.2262573
4: -125.4367523, 220.0620270, -130.3226624, 228.5426483, -353.9794006, 350.3846741

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3233651, upper bound: 188.3203480
time: 0.68 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3215552, upper bound: 188.3207773
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -53.2817039, 139.3572388, -51.0635910, 133.5844116, -186.8661194, 190.4208069
1: -128.1620789, 207.6663666, -122.8893051, 199.1136322, -327.2756958, 330.5556641
2: -87.1664886, 198.8413086, -83.5510254, 190.7286835, -277.8950500, 282.3923340
3: -139.5358276, 240.3794708, -133.8546753, 230.4699707, -370.0057983, 374.2341309
4: -130.5832977, 229.0112457, -125.1628265, 219.5740967, -350.1574097, 354.1740112

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3206433, upper bound: 188.3200435
time: 0.69 seconds

## Relational analysis of NS_A1_B1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3229833, upper bound: 188.3203884
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -53.2817039, 139.3572388, -53.1687355, 139.0787506, -192.3604584, 192.5259552
1: -128.1620789, 207.6663666, -127.8708878, 207.2600708, -335.4221191, 335.5372314
2: -87.1664886, 198.8413086, -86.9767456, 198.4295807, -285.5959473, 285.8180542
3: -139.5358276, 240.3794708, -139.2464447, 239.8921814, -379.4279785, 379.6259155
4: -130.5832977, 229.0112457, -130.3226624, 228.5426483, -359.1259460, 359.3338928

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B1_A2_B2_B1

### Relational analysis result of NS_A1_B1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3206433, upper bound: 188.3200435
time: 0.92 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_B2

### Relational analysis result of NS_A1_B1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3229833, upper bound: 188.3203884
time: 1.12 seconds

## BFS NS instance: NS_A1_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -51.0635910, 133.5844116, -56.9627800, 149.1831055, -200.2466888, 190.5471802
1: -122.8893051, 199.1136322, -135.9725952, 221.6457672, -344.5350342, 335.0862122
2: -83.5510254, 190.7286835, -92.7669983, 211.8950348, -295.4460144, 283.4955444
3: -133.8546753, 230.4699707, -149.0382233, 256.3016663, -390.1563416, 379.5081787
4: -125.1628265, 219.5740967, -140.0623627, 244.1065216, -369.2693481, 359.6364441

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B2_A1_A1_B1

### Relational analysis result of NS_A1_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3196946, upper bound: 188.3234748
time: 1.14 seconds

## Relational analysis of NS_A1_B1_B2_A1_A1_B2

### Relational analysis result of NS_A1_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3196946, upper bound: 188.3230089
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -53.1687355, 139.0787506, -57.4151039, 150.3148346, -203.4835663, 196.4938354
1: -127.8708878, 207.2600708, -137.0466309, 223.3119202, -351.1828003, 344.3066406
2: -86.9767456, 198.4295807, -93.4942780, 213.5201721, -300.4968567, 291.9237061
3: -139.2464447, 239.8921814, -150.1313629, 258.2296753, -397.4761353, 390.0235291
4: -130.3226624, 228.5426483, -141.1375580, 245.9391327, -376.2617798, 369.6801758

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B2_A1_A2_A1

### Relational analysis result of NS_A1_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2937584, upper bound: 188.3035672
time: 0.79 seconds

## Relational analysis of NS_A1_B1_B2_A1_A2_A2

### Relational analysis result of NS_A1_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2933691, upper bound: 188.3027499
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -55.8960342, 146.4871826, -56.9627800, 149.1831055, -205.0791321, 203.4499664
1: -133.2895660, 217.6700745, -135.9725952, 221.6457672, -354.9353333, 353.6426392
2: -90.9817352, 207.9409637, -92.7669983, 211.8950348, -302.8767700, 300.7079163
3: -146.2597046, 251.6436005, -149.0382233, 256.3016663, -402.5613708, 400.6818237
4: -137.5261230, 239.6225739, -140.0623627, 244.1065216, -381.6326294, 379.6849060

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3197422, upper bound: 188.3197422
time: 0.82 seconds

## Relational analysis of NS_A1_B1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3197422, upper bound: 188.3205367
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -58.6566620, 153.6555176, -57.4151039, 150.3148346, -208.9714661, 211.0706177
1: -139.9161377, 228.3240051, -137.0466309, 223.3119202, -363.2280579, 365.3705139
2: -95.5035553, 218.1519623, -93.4942780, 213.5201721, -309.0236511, 311.6462097
3: -153.3867950, 263.9873657, -150.1313629, 258.2296753, -411.6164551, 414.1187134
4: -144.2658386, 251.3736572, -141.1375580, 245.9391327, -390.2049561, 392.5111694

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3205367, upper bound: 188.3197420
time: 0.82 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3205367, upper bound: 188.3205367
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -55.6473885, 145.2252960, -55.5342064, 144.9453430, -200.5927277, 200.7594910
1: -135.1975555, 216.2953491, -134.8983002, 215.8840332, -351.0815735, 351.1936646
2: -91.3291931, 208.1940308, -91.1362610, 207.7673187, -299.0964966, 299.3302002
3: -145.8314514, 251.2343597, -145.5397491, 250.7280121, -396.5593872, 396.7741089
4: -135.7228546, 239.3147125, -135.4631958, 238.8403931, -374.5632324, 374.7778931

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_B1_B1_A1_B1

### Relational analysis result of NS_A2_B2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3009922, upper bound: 188.2887765
time: 1.31 seconds

## Relational analysis of NS_A2_B2_B1_B1_A1_B2

### Relational analysis result of NS_A2_B2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2999145, upper bound: 188.2883841
time: 1.18 seconds

## BFS NS instance: NS_A2_B2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -68.4621887, 178.0489502, -55.5342064, 144.9453430, -213.4075317, 233.5831604
1: -166.7131500, 264.8541870, -134.8983002, 215.8840332, -382.5971680, 399.7525024
2: -112.5068588, 255.3504944, -91.1362610, 207.7673187, -320.2741699, 346.4867249
3: -179.6618195, 307.1236572, -145.5397491, 250.7280121, -430.3897705, 452.6633911
4: -166.7109833, 293.4095764, -135.4631958, 238.8403931, -405.5513916, 428.8727722

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B1_B1_A2_A1

### Relational analysis result of NS_A2_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3020126, upper bound: 188.2902226
time: 1.28 seconds

## Relational analysis of NS_A2_B2_B1_B1_A2_A2

### Relational analysis result of NS_A2_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3020126, upper bound: 188.2902226
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -54.8618164, 143.0469360, -60.6216316, 157.7238770, -212.5856934, 203.6685638
1: -133.3801575, 213.1545715, -147.0804138, 234.7660980, -368.1462402, 360.2349548
2: -90.0561905, 205.3053131, -99.4746475, 225.9137421, -315.9699402, 304.7799072
3: -143.7186890, 247.8542328, -158.9473267, 272.1668091, -415.8854980, 406.8015747
4: -133.7547913, 235.8163452, -148.0004578, 259.6882629, -393.4430542, 383.8167419

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_B1_B2_B1_A1

### Relational analysis result of NS_A2_B2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2987077, upper bound: 188.2903066
time: 0.85 seconds

## Relational analysis of NS_A2_B2_B1_B2_B1_A2

### Relational analysis result of NS_A2_B2_B1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2708703, upper bound: 188.2592630
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -55.4394646, 144.3226166, -68.5210876, 178.1995239, -233.6389618, 212.8436890
1: -134.5274506, 215.0139771, -166.8324280, 265.1459656, -399.6733704, 381.8464050
2: -90.9789886, 207.0786285, -112.6328201, 255.5930176, -346.5720215, 319.7114563
3: -145.1796112, 249.8641205, -179.8773499, 307.4772339, -452.6567993, 429.7413635
4: -135.3091736, 237.8481598, -166.9021149, 293.6741638, -428.9833374, 404.7502747

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_B1_B2_B2_A1

### Relational analysis result of NS_A2_B2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2929700, upper bound: 188.2839889
time: 0.87 seconds

## Relational analysis of NS_A2_B2_B1_B2_B2_A2

### Relational analysis result of NS_A2_B2_B1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2675836, upper bound: 188.2552591
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -58.1504898, 151.5757751, -62.6828194, 163.6905975, -221.8410950, 214.2585754
1: -141.4646149, 225.7660065, -151.2567139, 243.1836243, -384.6482239, 377.0227051
2: -95.4863968, 217.5606689, -102.4806824, 233.7100830, -329.1964722, 320.0413513
3: -152.3531952, 262.4815369, -163.9895325, 282.3391724, -434.6922607, 426.4710693
4: -141.7097168, 249.8923035, -153.3971558, 268.6347046, -410.3444214, 403.2894592

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3042226, upper bound: 188.3124224
time: 1.35 seconds

## Relational analysis of NS_A2_B2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3042226, upper bound: 188.3168855
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -69.0855789, 180.4136505, -62.6655350, 163.6763611, -232.7619324, 243.0791931
1: -166.8268280, 267.8380432, -151.2021484, 243.1779633, -410.0047913, 419.0401917
2: -113.0089417, 257.3710938, -102.4563065, 233.6885071, -346.6974487, 359.8273621
3: -180.9311523, 310.2409973, -163.9427490, 282.3148193, -463.2459717, 474.1836853
4: -168.9928284, 296.0446777, -153.3584900, 268.6174927, -437.6103210, 449.4031677

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_B2_A2_A2_B1

### Relational analysis result of NS_A2_B2_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2858329, upper bound: 188.2858330
time: 1.15 seconds

## Relational analysis of NS_A2_B2_B2_A2_A2_B2

### Relational analysis result of NS_A2_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2858330, upper bound: 188.3042660
time: 1.14 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.12 seconds
NS_A1_B1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 0, lower bound: -188.3206434, upper bound: 188.3204136
NS_A1_B1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 0, lower bound: -188.3206434, upper bound: 188.3207773
NS_A1_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 0, lower bound: -188.3233651, upper bound: 188.3203480
NS_A1_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 0, lower bound: -188.3215552, upper bound: 188.3207773
NS_A1_B1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 0, lower bound: -188.3206433, upper bound: 188.3200435
NS_A1_B1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 0, lower bound: -188.3229833, upper bound: 188.3203884
NS_A1_B1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 0, lower bound: -188.3206433, upper bound: 188.3200435
NS_A1_B1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 0, lower bound: -188.3229833, upper bound: 188.3203884
NS_A1_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 0, lower bound: -188.3196946, upper bound: 188.3234748
NS_A1_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 0, lower bound: -188.3196946, upper bound: 188.3230089
NS_A1_B1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 0, lower bound: -188.2937584, upper bound: 188.3035672
NS_A1_B1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 0, lower bound: -188.2933691, upper bound: 188.3027499
NS_A1_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 0, lower bound: -188.3197422, upper bound: 188.3197422
NS_A1_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 0, lower bound: -188.3197422, upper bound: 188.3205367
NS_A1_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 0, lower bound: -188.3205367, upper bound: 188.3197420
NS_A1_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 0, lower bound: -188.3205367, upper bound: 188.3205367
NS_A2_B2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 0, lower bound: -188.3009922, upper bound: 188.2887765
NS_A2_B2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 0, lower bound: -188.2999145, upper bound: 188.2883841
NS_A2_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 0, lower bound: -188.3020126, upper bound: 188.2902226
NS_A2_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 0, lower bound: -188.3020126, upper bound: 188.2902226
NS_A2_B2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 0, lower bound: -188.2987077, upper bound: 188.2903066
NS_A2_B2_B1_B2_B1_A2, status: Status.VERIFIED, split count: 6, time: 3.12
Output dim: 0, lower bound: -188.2708703, upper bound: 188.2592630
NS_A2_B2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 6, time: 3.12
Output dim: 0, lower bound: -188.2929700, upper bound: 188.2839889
NS_A2_B2_B1_B2_B2_A2, status: Status.VERIFIED, split count: 6, time: 3.12
Output dim: 0, lower bound: -188.2675836, upper bound: 188.2552591
NS_A2_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 0, lower bound: -188.3042226, upper bound: 188.3124224
NS_A2_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 0, lower bound: -188.3042226, upper bound: 188.3168855
NS_A2_B2_B2_A2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.12
Output dim: 0, lower bound: -188.2858329, upper bound: 188.2858330
NS_A2_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 0, lower bound: -188.2858330, upper bound: 188.3042660

## BFS NS instance: NS_A1_B1_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -50.5746536, 132.3295593, -49.1869507, 128.8049164, -179.3795624, 181.5164795
1: -121.6970825, 197.2574615, -118.2904587, 192.0524445, -313.7494812, 315.5479126
2: -82.7451706, 188.9421997, -80.4527893, 183.8756409, -266.6207886, 269.3949890
3: -132.5543518, 228.3417664, -128.9042206, 222.3095856, -354.8639526, 357.2459717
4: -123.9665909, 217.5167542, -120.6151276, 211.7019501, -335.6685486, 338.1318970

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B1_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3214732, upper bound: 188.3209902
time: 0.76 seconds

## Relational analysis of NS_A1_B1_B1_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3214732, upper bound: 188.3210849
time: 1.03 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -50.2992172, 131.6348114, -50.3314896, 132.2662048, -182.5654144, 181.9662933
1: -121.0290375, 196.2342529, -121.2628174, 197.3463745, -318.3753357, 317.4970093
2: -82.2957687, 187.9580383, -82.3782425, 188.9986420, -271.2943726, 270.3362732
3: -131.8150787, 227.1602631, -131.9352112, 228.5007629, -360.3158264, 359.0954590
4: -123.2857208, 216.3789825, -123.3338852, 217.6429291, -340.9286194, 339.7127991

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B1_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3217476, upper bound: 188.3209902
time: 0.82 seconds

## Relational analysis of NS_A1_B1_B1_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3217476, upper bound: 188.3210849
time: 1.12 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -49.3013191, 129.0867615, -52.5424004, 137.4888306, -186.7901154, 181.6291504
1: -118.5803909, 192.4651642, -126.3357773, 204.9140778, -323.4944763, 318.8009033
2: -80.6434937, 184.2910614, -85.9435196, 196.1496429, -276.7931213, 270.2345581
3: -129.1960144, 222.8027649, -137.5966797, 237.1775360, -366.3735352, 360.3994446
4: -120.8805008, 212.1748047, -128.8043213, 225.9226837, -346.8031311, 340.9791260

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3206434, upper bound: 188.3203480
time: 0.81 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3206434, upper bound: 188.3203480
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -50.4469948, 132.5516663, -52.3181000, 136.9186249, -187.3656158, 184.8697662
1: -121.5557861, 197.7647705, -125.7931671, 204.0658112, -325.6215820, 323.5579224
2: -82.5709381, 189.4189606, -85.5776749, 195.3484344, -277.9193420, 274.9966125
3: -132.2305145, 229.0004425, -136.9888611, 236.2077484, -368.4382324, 365.9892578
4: -123.6018753, 218.1224823, -128.2476959, 224.9948425, -348.5967102, 346.3700562

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3206434, upper bound: 188.3204136
time: 0.82 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3206434, upper bound: 188.3207773
time: 1.13 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -52.6556702, 137.7680359, -49.1869507, 128.8049164, -181.4605865, 186.9549408
1: -126.6277618, 205.3213348, -118.2904587, 192.0524445, -318.6801758, 323.6117859
2: -86.1336212, 196.5624084, -80.4527893, 183.8756409, -270.0092468, 277.0151978
3: -137.8866577, 237.6660309, -128.9042206, 222.3095856, -360.1961670, 366.5702515
4: -129.0654907, 226.3925934, -120.6151276, 211.7019501, -340.7674561, 347.0077209

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B1_A2_B1_B1_A1

### Relational analysis result of NS_A1_B1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3213654, upper bound: 188.3196946
time: 0.77 seconds

## Relational analysis of NS_A1_B1_B1_A2_B1_B1_A2

### Relational analysis result of NS_A1_B1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3213654, upper bound: 188.3202363
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -52.4316978, 137.1985931, -50.3314896, 132.2662048, -184.6979065, 187.5300751
1: -126.0864792, 204.4739838, -121.2628174, 197.3463745, -323.4327698, 325.7367554
2: -85.7684250, 195.7619324, -82.3782425, 188.9986420, -274.7670593, 278.1401672
3: -137.2800446, 236.6974640, -131.9352112, 228.5007629, -365.7808228, 368.6326904
4: -128.5094147, 225.4659729, -123.3338852, 217.6429291, -346.1523438, 348.7998352

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B1_A2_B1_B2_A1

### Relational analysis result of NS_A1_B1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3230089, upper bound: 188.3196946
time: 0.70 seconds

## Relational analysis of NS_A1_B1_B1_A2_B1_B2_A2

### Relational analysis result of NS_A1_B1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3217476, upper bound: 188.3203884
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -52.6556702, 137.7680359, -51.2118454, 134.1105347, -186.7662048, 188.9798737
1: -126.6277618, 205.3213348, -123.0768661, 199.9316559, -326.5594177, 328.3981934
2: -86.1336212, 196.5624084, -83.7500458, 191.3062286, -277.4398193, 280.3124390
3: -137.8866577, 237.6660309, -134.0930023, 231.4107819, -369.2974243, 371.7590332
4: -129.0654907, 226.3925934, -125.5799866, 220.3556671, -349.4211426, 351.9725952

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B1_A2_B2_B1_A1

### Relational analysis result of NS_A1_B1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3206433, upper bound: 188.3196946
time: 1.32 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_B1_A2

### Relational analysis result of NS_A1_B1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3206433, upper bound: 188.3200435
time: 0.94 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -52.4316978, 137.1985931, -52.5543442, 138.0796509, -190.5113525, 189.7529297
1: -126.0864792, 204.4739838, -126.5314484, 205.9635620, -332.0499878, 331.0054321
2: -85.7684250, 195.7619324, -85.9974899, 197.1540833, -282.9224854, 281.7593994
3: -137.2800446, 236.6974640, -137.6338043, 238.4583893, -375.7384338, 374.3312378
4: -128.5094147, 225.4659729, -128.7805939, 227.1076202, -355.6170349, 354.2465820

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B1_A2_B2_B2_A1

### Relational analysis result of NS_A1_B1_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3229833, upper bound: 188.3196946
time: 0.77 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_B2_A2

### Relational analysis result of NS_A1_B1_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3229833, upper bound: 188.3203884
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -50.4570312, 132.0393524, -55.0949745, 144.4189758, -194.8760071, 187.1343231
1: -121.4000092, 196.8309479, -131.3846741, 214.6123047, -336.0123291, 328.2155151
2: -82.5493240, 188.5142212, -89.6710587, 205.0620880, -287.6113892, 278.1852112
3: -132.2541351, 227.8324127, -144.1025848, 248.1752319, -380.4293213, 371.9349976
4: -123.6929092, 217.0294189, -135.5324860, 236.2595062, -359.9523926, 352.5618896

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B2_A1_A1_B1_A1

### Relational analysis result of NS_A1_B1_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3196946, upper bound: 188.3213654
time: 0.71 seconds

## Relational analysis of NS_A1_B1_B2_A1_A1_B1_A2

### Relational analysis result of NS_A1_B1_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3196946, upper bound: 188.3230089
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -50.1819038, 131.3451843, -56.3468437, 148.2128601, -198.3947296, 187.6920166
1: -120.7323151, 195.8085175, -134.6346283, 220.4153748, -341.1476440, 330.4431458
2: -82.1003799, 187.5309296, -91.7802887, 210.6768036, -292.7771301, 279.3111877
3: -131.5155029, 226.6521301, -147.4225769, 254.9489594, -386.4644775, 374.0747070
4: -123.0126648, 215.8925629, -138.5203094, 242.7539673, -365.7666016, 354.4128723

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B2_A1_A1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3202366, upper bound: 188.3213654
time: 0.72 seconds

## Relational analysis of NS_A1_B1_B2_A1_A1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3202366, upper bound: 188.3230089
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_B2_A1_A2_A1

### Backsubstitution after applying NS history:
0: -52.0760078, 136.3224640, -56.9953575, 149.2497253, -201.3257294, 193.3178253
1: -125.1260605, 203.1962280, -135.9845428, 221.7375183, -346.8635559, 339.1807861
2: -85.1593323, 194.4136047, -92.7930145, 211.9590607, -297.1184082, 287.2065735
3: -136.4144592, 235.1133118, -149.0379791, 256.3831787, -392.7976379, 384.1513062
4: -127.7403793, 223.9375916, -140.1457062, 244.1672516, -371.9076233, 364.0832520

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B2_A1_A2_A1_B1

### Relational analysis result of NS_A1_B1_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2933691, upper bound: 188.3027499
time: 0.71 seconds

## Relational analysis of NS_A1_B1_B2_A1_A2_A1_B2

### Relational analysis result of NS_A1_B1_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2933691, upper bound: 188.3027499
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_B2_A1_A2_A2

### Backsubstitution after applying NS history:
0: -53.1297150, 138.4694672, -55.7253418, 145.7905426, -198.9202576, 194.1948090
1: -127.6821823, 206.4684448, -132.8463287, 216.6410675, -344.3232422, 339.3147583
2: -86.9134903, 197.5112000, -90.6851883, 206.9882050, -293.9016724, 288.1963196
3: -139.1647644, 238.8140259, -145.6951599, 250.3909149, -389.5556641, 384.5091858
4: -130.3755493, 227.5046844, -137.0917664, 238.4697113, -368.8452759, 364.5964355

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B2_A1_A2_A2_B1

### Relational analysis result of NS_A1_B1_B2_A1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2694488, upper bound: 188.2817372
time: 0.82 seconds

## Relational analysis of NS_A1_B1_B2_A1_A2_A2_B2

### Relational analysis result of NS_A1_B1_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2915787, upper bound: 188.3009966
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -55.8960342, 146.4871826, -55.8960342, 146.4871826, -202.3831940, 202.3832092
1: -133.2895660, 217.6700745, -133.2895660, 217.6700745, -350.9596252, 350.9596252
2: -90.9817352, 207.9409637, -90.9817352, 207.9409637, -298.9226990, 298.9226990
3: -146.2597046, 251.6436005, -146.2597046, 251.6436005, -397.9033203, 397.9033203
4: -137.5261230, 239.6225739, -137.5261230, 239.6225739, -377.1486816, 377.1486816

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B2_A2_A1_B1_B1

### Relational analysis result of NS_A1_B1_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3160869, upper bound: 188.3191378
time: 1.24 seconds

## Relational analysis of NS_A1_B1_B2_A2_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3174370, upper bound: 188.3190748
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -55.8960342, 146.4871826, -58.6566620, 153.6555176, -209.5515442, 205.1438141
1: -133.2895660, 217.6700745, -139.9161377, 228.3240051, -361.6134949, 357.5862122
2: -90.9817352, 207.9409637, -95.5035553, 218.1519623, -309.1336975, 303.4444885
3: -146.2597046, 251.6436005, -153.3867950, 263.9873657, -410.2470703, 405.0303955
4: -137.5261230, 239.6225739, -144.2658386, 251.3736572, -388.8997192, 383.8884277

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B2_A2_A1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3174240, upper bound: 188.3190752
time: 0.69 seconds

## Relational analysis of NS_A1_B1_B2_A2_A1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3174370, upper bound: 188.3190880
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -58.6566620, 153.6555176, -55.8960342, 146.4871826, -205.1437988, 209.5515442
1: -139.9161377, 228.3240051, -133.2895660, 217.6700745, -357.5862122, 361.6135254
2: -95.5035553, 218.1519623, -90.9817352, 207.9409637, -303.4445190, 309.1336975
3: -153.3867950, 263.9873657, -146.2597046, 251.6436005, -405.0303955, 410.2470703
4: -144.2658386, 251.3736572, -137.5261230, 239.6225739, -383.8884277, 388.8997192

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_B2_A2_A2_B1_A1

### Relational analysis result of NS_A1_B1_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3174288, upper bound: 188.3121574
time: 0.73 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2_B1_A2

### Relational analysis result of NS_A1_B1_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3203012, upper bound: 188.3196241
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -58.6566620, 153.6555176, -58.6566620, 153.6555176, -212.3121490, 212.3121643
1: -139.9161377, 228.3240051, -139.9161377, 228.3240051, -368.2401123, 368.2401123
2: -95.5035553, 218.1519623, -95.5035553, 218.1519623, -313.6554871, 313.6554871
3: -153.3867950, 263.9873657, -153.3867950, 263.9873657, -417.3741455, 417.3741455
4: -144.2658386, 251.3736572, -144.2658386, 251.3736572, -395.6394653, 395.6394653

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_A1

### Relational analysis result of NS_A1_B1_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3174240, upper bound: 188.3160869
time: 0.78 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3183126, upper bound: 188.3174370
time: 1.24 seconds

## BFS NS instance: NS_A2_B2_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -55.5726967, 145.0267639, -55.0474586, 143.6520233, -199.2247162, 200.0742035
1: -135.0212555, 216.0026855, -133.7488556, 213.9770203, -348.9982300, 349.7515259
2: -91.2088470, 207.9209595, -90.3518219, 205.9875641, -297.1963806, 298.2726746
3: -145.6343384, 250.9025421, -144.2554321, 248.5650482, -394.1994019, 395.1579590
4: -135.5386505, 238.9940338, -134.2627716, 236.7511292, -372.2897949, 373.2568054

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3009922, upper bound: 188.2887765
time: 0.83 seconds

## Relational analysis of NS_A2_B2_B1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3009922, upper bound: 188.2887765
time: 0.99 seconds

## BFS NS instance: NS_A2_B2_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -55.4515228, 144.7191467, -55.0080452, 143.5851898, -199.0366974, 199.7271881
1: -134.7313385, 215.5498810, -133.6588440, 213.8830109, -348.6143494, 349.2086487
2: -91.0130005, 207.4833069, -90.2967758, 205.8743744, -296.8873901, 297.7800903
3: -145.3152313, 250.3785248, -144.1490173, 248.4380493, -393.7532654, 394.5275269
4: -135.2441559, 238.4917450, -134.1679688, 236.6396179, -371.8837891, 372.6597290

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3003059, upper bound: 188.2883841
time: 0.86 seconds

## Relational analysis of NS_A2_B2_B1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3003059, upper bound: 188.2883841
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -68.3880310, 177.8731689, -55.5342064, 144.9453430, -213.3333740, 233.4073792
1: -166.5200500, 264.5990906, -134.8983002, 215.8840332, -382.4040833, 399.4973755
2: -112.3821106, 255.0892029, -91.1362610, 207.7673187, -320.1494141, 346.2254639
3: -179.4766388, 306.8139648, -145.5397491, 250.7280121, -430.2046509, 452.3536987
4: -166.5419922, 293.1131897, -135.4631958, 238.8403931, -405.3823853, 428.5763855

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_B1_B1_A2_A1_B1

### Relational analysis result of NS_A2_B2_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3015134, upper bound: 188.2893055
time: 0.86 seconds

## Relational analysis of NS_A2_B2_B1_B1_A2_A1_B2

### Relational analysis result of NS_A2_B2_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.2999145, upper bound: 188.2885712
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -69.0842972, 180.4100800, -55.5342064, 144.9453430, -214.0296326, 235.9442902
1: -166.8239746, 267.8328247, -134.8983002, 215.8840332, -382.7079468, 402.7311401
2: -113.0069199, 257.3662415, -91.1362610, 207.7673187, -320.7742310, 348.5024719
3: -180.9277191, 310.2351379, -145.5397491, 250.7280121, -431.6557007, 455.7749023
4: -168.9895172, 296.0390015, -135.4631958, 238.8403931, -407.8298950, 431.5021973

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_B1_B1_A2_A2_B1

### Relational analysis result of NS_A2_B2_B1_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2908743, upper bound: 188.2783936
time: 0.80 seconds

## Relational analysis of NS_A2_B2_B1_B1_A2_A2_B2

### Relational analysis result of NS_A2_B2_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -188.3019677, upper bound: 188.2896667
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -52.4689751, 136.8182678, -59.9174728, 155.8513641, -208.3203430, 196.7356873
1: -127.4188538, 203.9490204, -145.3468933, 231.9822235, -359.4010620, 349.2958679
2: -86.0892868, 196.4375916, -98.3166351, 223.2612610, -309.3504944, 294.7542114
3: -137.3227844, 237.2638092, -157.0745087, 268.9850159, -406.3078003, 394.3382568
4: -127.9355850, 225.5622253, -146.2890472, 256.6081543, -384.5437012, 371.8512573

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B1_B2_B1_A1_A1

### Relational analysis result of NS_A2_B2_B1_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2816385, upper bound: 188.2728271
time: 1.21 seconds

## Relational analysis of NS_A2_B2_B1_B2_B1_A1_A2

### Relational analysis result of NS_A2_B2_B1_B2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -188.2735847, upper bound: 188.2623958
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -57.5952988, 150.0659180, -62.6828194, 163.6905975, -221.2858887, 212.7487335
1: -140.1770020, 223.6007080, -151.2567139, 243.1836243, -383.3606262, 374.8574219
2: -94.5957947, 215.3483429, -102.4806824, 233.7100830, -328.3058777, 317.8290405
3: -150.9304047, 260.0450439, -163.9895325, 282.3391724, -433.2695312, 424.0345764
4: -140.3708038, 247.4193115, -153.3971558, 268.6347046, -409.0054932, 400.8164673

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 33

## BFS NS instance: NS_A2_B2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -57.9521370, 151.0606842, -62.6828194, 163.6905975, -221.6427307, 213.7434692
1: -140.9779968, 224.9926758, -151.2567139, 243.1836243, -384.1615601, 376.2493896
2: -95.1560898, 216.8126221, -102.4806824, 233.7100830, -328.8661804, 319.2932434
3: -151.8334961, 261.5792542, -163.9895325, 282.3391724, -434.1725464, 425.5687866
4: -141.2284088, 249.0352325, -153.3971558, 268.6347046, -409.8630981, 402.4323730

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

## BFS NS instance: NS_A2_B2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -69.0855789, 180.4136505, -69.0855789, 180.4136505, -249.4992218, 249.4992218
1: -166.8268280, 267.8380432, -166.8268280, 267.8380432, -434.6648560, 434.6648560
2: -113.0089417, 257.3710938, -113.0089417, 257.3710938, -370.3800354, 370.3800354
3: -180.9311523, 310.2409973, -180.9311523, 310.2409973, -491.1721191, 491.1721191
4: -168.9928284, 296.0446777, -168.9928284, 296.0446777, -465.0375061, 465.0375061

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.87 + 214.30 = 217.18 seconds
