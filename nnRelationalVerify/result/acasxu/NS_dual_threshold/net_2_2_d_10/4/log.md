## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 339.41632513516


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-149.8139038, 226.3787994, -149.8139038, 226.3787994, -376.1926880, 376.1926880)
1: (-118.9682159, 218.5465088, -118.9682159, 218.5465088, -337.5147095, 337.5147095)
2: (-102.7016296, 224.2848206, -102.7016296, 224.2848206, -326.9864502, 326.9864502)
3: (-155.0733032, 221.6566010, -155.0733032, 221.6566010, -376.7298584, 376.7298584)
4: (-123.6448669, 237.1756439, -123.6448669, 237.1756439, -360.8204956, 360.8204956)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.44 + 2.07 = 3.51 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -339.4332968, upper bound: 339.4332968

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4272639, upper bound: 339.4332968
time: 0.98 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4272639, upper bound: 339.4274662
time: 1.08 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.20 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 2.20
Output dim: 4, lower bound: -339.4272639, upper bound: 339.4332968
NS_B2, status: Status.UNKNOWN, split count: 1, time: 2.20
Output dim: 4, lower bound: -339.4272639, upper bound: 339.4274662

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -149.8139038, 226.3787994, -139.2357330, 210.1520996, -359.9660034, 365.6145325
1: -118.9682159, 218.5465088, -110.5789261, 202.7296143, -321.6978149, 329.1254272
2: -102.7016296, 224.2848206, -95.4418259, 208.1913605, -310.8930054, 319.7266235
3: -155.0733032, 221.6566010, -144.2740173, 205.6983643, -360.7716675, 365.9305420
4: -123.6448669, 237.1756439, -114.9057312, 220.2528229, -343.8977051, 352.0813599

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4271323, upper bound: 339.4257748
time: 0.85 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4254409, upper bound: 339.4257748
time: 0.80 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -149.8139038, 226.3787994, -148.5892334, 223.2648010, -373.0787048, 374.9680176
1: -118.9682159, 218.5465088, -117.4431000, 215.2394714, -334.2076416, 335.9896240
2: -102.7016296, 224.2848206, -101.5047607, 221.2780457, -323.9796753, 325.7895813
3: -155.0733032, 221.6566010, -153.3717346, 218.3774719, -373.4507751, 375.0283203
4: -123.6448669, 237.1756439, -122.3112946, 234.1312103, -357.7760620, 359.4869080

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4233447, upper bound: 339.4193716
time: 1.48 seconds

## Relational analysis of NS_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4274662, upper bound: 339.4257748
time: 1.00 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4254409, upper bound: 339.4257748
time: 0.97 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.38 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 5.38
Output dim: 4, lower bound: -339.4271323, upper bound: 339.4257748
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 5.38
Output dim: 4, lower bound: -339.4254409, upper bound: 339.4257748
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 5.38
Output dim: 4, lower bound: -339.4274662, upper bound: 339.4257748
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 5.38
Output dim: 4, lower bound: -339.4254409, upper bound: 339.4257748

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -142.6678162, 215.3473511, -139.2357330, 210.1520996, -352.8199158, 354.5830688
1: -113.2607117, 207.5004425, -110.5789261, 202.7296143, -315.9903259, 318.0793762
2: -97.7639313, 213.1726685, -95.4418259, 208.1913605, -305.9552307, 308.6145020
3: -147.7779846, 210.6499481, -144.2740173, 205.6983643, -353.4763489, 354.9239502
4: -117.6859665, 225.4986725, -114.9057312, 220.2528229, -337.9387817, 340.4044189

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B1_A1_B1

### Relational analysis result of NS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4254409, upper bound: 339.4257748
time: 0.90 seconds

## Relational analysis of NS_B1_A1_B2

### Relational analysis result of NS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4254409, upper bound: 339.4257748
time: 1.04 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -190.1032867, 283.8924866, -138.5508118, 209.1209564, -399.2241821, 422.4432983
1: -149.7224426, 272.0595703, -110.0489426, 201.7363281, -351.4586182, 382.1084290
2: -129.3840485, 280.4438477, -94.9811707, 207.1735077, -336.5574646, 375.4250183
3: -195.3681641, 276.7221069, -143.5873718, 204.6928711, -400.0609436, 420.3094482
4: -156.0772247, 296.6318970, -114.3495331, 219.1790619, -375.2562866, 410.9813538

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4254409, upper bound: 339.4257748
time: 0.65 seconds

## Relational analysis of NS_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4254409, upper bound: 339.4257748
time: 0.75 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -142.6678162, 215.3473511, -148.5892334, 223.2648010, -365.9326172, 363.9365845
1: -113.2607117, 207.5004425, -117.4431000, 215.2394714, -328.5001526, 324.9435425
2: -97.7639313, 213.1726685, -101.5047607, 221.2780457, -319.0419922, 314.6774292
3: -147.7779846, 210.6499481, -153.3717346, 218.3774719, -366.1554565, 364.0216675
4: -117.6859665, 225.4986725, -122.3112946, 234.1312103, -351.8171692, 347.8099365

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_A1_B1

### Relational analysis result of NS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4254409, upper bound: 339.4257748
time: 0.63 seconds

## Relational analysis of NS_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4257748, upper bound: 339.4257748
time: 1.07 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -190.1032867, 283.8924866, -147.8494110, 222.1550751, -412.2583618, 431.7418823
1: -149.7224426, 272.0595703, -116.8687515, 214.1746368, -363.8970032, 388.9282532
2: -129.3840485, 280.4438477, -101.0050659, 220.1878967, -349.5719299, 381.4489136
3: -195.3681641, 276.7221069, -152.6323090, 217.2956696, -412.6638184, 429.3543701
4: -156.0772247, 296.6318970, -121.7033920, 232.9811401, -389.0583496, 418.3352356

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_A2_B1

### Relational analysis result of NS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4257748, upper bound: 339.4257748
time: 0.89 seconds

## Relational analysis of NS_B2_A2_B2

### Relational analysis result of NS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4257748, upper bound: 339.4257748
time: 1.02 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.40 seconds
NS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 4, lower bound: -339.4254409, upper bound: 339.4257748
NS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 4, lower bound: -339.4254409, upper bound: 339.4257748
NS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 4, lower bound: -339.4254409, upper bound: 339.4257748
NS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 4, lower bound: -339.4254409, upper bound: 339.4257748
NS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 4, lower bound: -339.4254409, upper bound: 339.4257748
NS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 4, lower bound: -339.4257748, upper bound: 339.4257748
NS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 4, lower bound: -339.4257748, upper bound: 339.4257748
NS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 4, lower bound: -339.4257748, upper bound: 339.4257748

## BFS NS instance: NS_B1_A1_B1

### Backsubstitution after applying NS history:
0: -142.6678162, 215.3473511, -132.0075226, 198.9833984, -341.6512146, 347.3548584
1: -113.2607117, 207.5004425, -104.8025665, 191.5839996, -304.8446960, 312.3029785
2: -97.7639313, 213.1726685, -90.4448700, 196.9623718, -294.7262573, 303.6175537
3: -147.7779846, 210.6499481, -136.8857422, 194.5722961, -342.3502808, 347.5357056
4: -117.6859665, 225.4986725, -108.8758087, 208.4512329, -326.1372070, 334.3744507

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B1_A1_B1_A1

### Relational analysis result of NS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4271323, upper bound: 339.4255725
time: 1.00 seconds

## Relational analysis of NS_B1_A1_B1_A2

### Relational analysis result of NS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4271323, upper bound: 339.4257748
time: 0.66 seconds

## BFS NS instance: NS_B1_A1_B2

### Backsubstitution after applying NS history:
0: -142.6678162, 215.3473511, -180.1691589, 268.6386108, -411.3064270, 395.5165100
1: -113.2607117, 207.5004425, -141.8278961, 257.1098022, -370.3705139, 349.3283386
2: -97.7639313, 213.1726685, -122.5468063, 265.2631531, -363.0270691, 335.7194824
3: -147.7779846, 210.6499481, -185.2196198, 261.6675415, -409.4455261, 395.8695679
4: -117.6859665, 225.4986725, -147.8655090, 280.6687927, -398.3547363, 373.3641357

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B1_A1_B2_A1

### Relational analysis result of NS_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4271323, upper bound: 339.4255725
time: 0.82 seconds

## Relational analysis of NS_B1_A1_B2_A2

### Relational analysis result of NS_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4271323, upper bound: 339.4257748
time: 0.87 seconds

## BFS NS instance: NS_B1_A2_B1

### Backsubstitution after applying NS history:
0: -190.1032867, 283.8924866, -132.0075226, 198.9833984, -389.0866699, 415.9000244
1: -149.7224426, 272.0595703, -104.8025665, 191.5839996, -341.3063965, 376.8620605
2: -129.3840485, 280.4438477, -90.4448700, 196.9623718, -326.3464355, 370.8886719
3: -195.3681641, 276.7221069, -136.8857422, 194.5722961, -389.9403992, 413.6078186
4: -156.0772247, 296.6318970, -108.8758087, 208.4512329, -364.5284424, 405.5076599

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B1_B1

### Relational analysis result of NS_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4212693, upper bound: 339.4225293
time: 0.94 seconds

## Relational analysis of NS_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B1_A2_B1_A1

### Relational analysis result of NS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4254409, upper bound: 339.4254409
time: 1.04 seconds

## Relational analysis of NS_B1_A2_B1_A2

### Relational analysis result of NS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4254409, upper bound: 339.4257748
time: 0.91 seconds

## BFS NS instance: NS_B1_A2_B2

### Backsubstitution after applying NS history:
0: -190.1032867, 283.8924866, -180.1691589, 268.6386108, -458.7418823, 464.0616455
1: -149.7224426, 272.0595703, -141.8278961, 257.1098022, -406.8322144, 413.8874207
2: -129.3840485, 280.4438477, -122.5468063, 265.2631531, -394.6472168, 402.9906006
3: -195.3681641, 276.7221069, -185.2196198, 261.6675415, -457.0356750, 461.9416809
4: -156.0772247, 296.6318970, -147.8655090, 280.6687927, -436.7460327, 444.4973450

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B1_A2_B2_A1

### Relational analysis result of NS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4254409, upper bound: 339.4254409
time: 1.02 seconds

## Relational analysis of NS_B1_A2_B2_A2

### Relational analysis result of NS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4254409, upper bound: 339.4257748
time: 0.88 seconds

## BFS NS instance: NS_B2_A1_B1

### Backsubstitution after applying NS history:
0: -142.6678162, 215.3473511, -141.9029388, 212.8578644, -355.5256958, 357.2502747
1: -113.2607117, 207.5004425, -112.0533905, 204.8308868, -318.0916138, 319.5538330
2: -97.7639313, 213.1726685, -96.8430252, 210.8139038, -308.5777893, 310.0156860
3: -147.7779846, 210.6499481, -146.4658966, 207.9918671, -355.7698059, 357.1158447
4: -117.6859665, 225.4986725, -116.6867065, 223.1286774, -340.8146362, 342.1853027

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B2_A1_B1_A1

### Relational analysis result of NS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4271323, upper bound: 339.4255725
time: 0.98 seconds

## Relational analysis of NS_B2_A1_B1_A2

### Relational analysis result of NS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4271323, upper bound: 339.4257748
time: 0.93 seconds

## BFS NS instance: NS_B2_A1_B2

### Backsubstitution after applying NS history:
0: -142.6678162, 215.3473511, -200.1334076, 297.7181091, -440.3859253, 415.4807739
1: -113.2607117, 207.5004425, -157.2297211, 284.7256775, -397.9863892, 364.7301636
2: -97.7639313, 213.1726685, -135.9115906, 293.9135132, -391.6773682, 349.0842590
3: -147.7779846, 210.6499481, -205.4288025, 289.8867188, -437.6647034, 416.0787354
4: -117.6859665, 225.4986725, -164.0981445, 310.9541931, -428.6401672, 389.5967712

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B2_A1_B2_A1

### Relational analysis result of NS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4271323, upper bound: 339.4255725
time: 0.65 seconds

## Relational analysis of NS_B2_A1_B2_A2

### Relational analysis result of NS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4271323, upper bound: 339.4257748
time: 0.94 seconds

## BFS NS instance: NS_B2_A2_B1

### Backsubstitution after applying NS history:
0: -190.1032867, 283.8924866, -141.9029388, 212.8578644, -402.9611511, 425.7954102
1: -149.7224426, 272.0595703, -112.0533905, 204.8308868, -354.5533447, 384.1129456
2: -129.3840485, 280.4438477, -96.8430252, 210.8139038, -340.1979370, 377.2868347
3: -195.3681641, 276.7221069, -146.4658966, 207.9918671, -403.3599243, 423.1879883
4: -156.0772247, 296.6318970, -116.6867065, 223.1286774, -379.2059021, 413.3185425

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_A2_B1_A1

### Relational analysis result of NS_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4149867, upper bound: 339.4112631
time: 0.69 seconds

## Relational analysis of NS_B2_A2_B1_A2

### Relational analysis result of NS_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4135167, upper bound: 339.4135167
time: 1.06 seconds

## BFS NS instance: NS_B2_A2_B2

### Backsubstitution after applying NS history:
0: -190.1032867, 283.8924866, -200.1334076, 297.7181091, -487.8213806, 484.0258789
1: -149.7224426, 272.0595703, -157.2297211, 284.7256775, -434.4481201, 429.2893066
2: -129.3840485, 280.4438477, -135.9115906, 293.9135132, -423.2975464, 416.3554382
3: -195.3681641, 276.7221069, -205.4288025, 289.8867188, -485.2548523, 482.1509094
4: -156.0772247, 296.6318970, -164.0981445, 310.9541931, -467.0314331, 460.7299805

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B2_A2_B2_A1

### Relational analysis result of NS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4254409, upper bound: 339.4254409
time: 1.05 seconds

## Relational analysis of NS_B2_A2_B2_A2

### Relational analysis result of NS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4254409, upper bound: 339.4257748
time: 1.03 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.59 seconds
NS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.59
Output dim: 4, lower bound: -339.4271323, upper bound: 339.4255725
NS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.59
Output dim: 4, lower bound: -339.4271323, upper bound: 339.4257748
NS_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.59
Output dim: 4, lower bound: -339.4271323, upper bound: 339.4255725
NS_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.59
Output dim: 4, lower bound: -339.4271323, upper bound: 339.4257748
NS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.59
Output dim: 4, lower bound: -339.4254409, upper bound: 339.4254409
NS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.59
Output dim: 4, lower bound: -339.4254409, upper bound: 339.4257748
NS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.59
Output dim: 4, lower bound: -339.4254409, upper bound: 339.4254409
NS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.59
Output dim: 4, lower bound: -339.4254409, upper bound: 339.4257748
NS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.59
Output dim: 4, lower bound: -339.4271323, upper bound: 339.4255725
NS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.59
Output dim: 4, lower bound: -339.4271323, upper bound: 339.4257748
NS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.59
Output dim: 4, lower bound: -339.4271323, upper bound: 339.4255725
NS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.59
Output dim: 4, lower bound: -339.4271323, upper bound: 339.4257748
NS_B2_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 3.59
Output dim: 4, lower bound: -339.4149867, upper bound: 339.4112631
NS_B2_A2_B1_A2, status: Status.VERIFIED, split count: 4, time: 3.59
Output dim: 4, lower bound: -339.4135167, upper bound: 339.4135167
NS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.59
Output dim: 4, lower bound: -339.4254409, upper bound: 339.4254409
NS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.59
Output dim: 4, lower bound: -339.4254409, upper bound: 339.4257748

## BFS NS instance: NS_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -132.0075226, 198.9833984, -132.0075226, 198.9833984, -330.9909058, 330.9909058
1: -104.8025665, 191.5839996, -104.8025665, 191.5839996, -296.3865051, 296.3865051
2: -90.4448700, 196.9623718, -90.4448700, 196.9623718, -287.4072266, 287.4072266
3: -136.8857422, 194.5722961, -136.8857422, 194.5722961, -331.4579773, 331.4579773
4: -108.8758087, 208.4512329, -108.8758087, 208.4512329, -317.3270264, 317.3270264

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_A1_B1_A1_A1

### Relational analysis result of NS_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4164033, upper bound: 339.4178401
time: 0.73 seconds

## Relational analysis of NS_B1_A1_B1_A1_A2

### Relational analysis result of NS_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4169693, upper bound: 339.4182115
time: 0.74 seconds

## BFS NS instance: NS_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -141.8882599, 212.8376312, -132.0075226, 198.9833984, -340.8716431, 344.8451233
1: -112.0422058, 204.8112488, -104.8025665, 191.5839996, -303.6261597, 309.6137390
2: -96.8333588, 210.7936554, -90.4448700, 196.9623718, -293.7956848, 301.2384644
3: -146.4514313, 207.9716187, -136.8857422, 194.5722961, -341.0236511, 344.8573608
4: -116.6749344, 223.1072540, -108.8758087, 208.4512329, -325.1261597, 331.9830627

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_A1_B1_A2_B1

### Relational analysis result of NS_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4174798, upper bound: 339.4251716
time: 0.79 seconds

## Relational analysis of NS_B1_A1_B1_A2_B2

### Relational analysis result of NS_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4169693, upper bound: 339.4206701
time: 0.90 seconds

## BFS NS instance: NS_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -132.0075226, 198.9833984, -180.1691589, 268.6386108, -400.6461182, 379.1525574
1: -104.8025665, 191.5839996, -141.8278961, 257.1098022, -361.9123230, 333.4118652
2: -90.4448700, 196.9623718, -122.5468063, 265.2631531, -355.7080078, 319.5091553
3: -136.8857422, 194.5722961, -185.2196198, 261.6675415, -398.5532837, 379.7918396
4: -108.8758087, 208.4512329, -147.8655090, 280.6687927, -389.5445557, 356.3167419

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_B2_A1_A1

### Relational analysis result of NS_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4260399, upper bound: 339.4236125
time: 0.87 seconds

## Relational analysis of NS_B1_A1_B2_A1_A2

### Relational analysis result of NS_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4266834, upper bound: 339.4251693
time: 0.76 seconds

## BFS NS instance: NS_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -141.8882599, 212.8376312, -180.1691589, 268.6386108, -410.5268555, 393.0067749
1: -112.0422058, 204.8112488, -141.8278961, 257.1098022, -369.1519775, 346.6391296
2: -96.8333588, 210.7936554, -122.5468063, 265.2631531, -362.0964966, 333.3404236
3: -146.4514313, 207.9716187, -185.2196198, 261.6675415, -408.1189270, 393.1912231
4: -116.6749344, 223.1072540, -147.8655090, 280.6687927, -397.3437195, 370.9727478

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_B2_A2_A1

### Relational analysis result of NS_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4260399, upper bound: 339.4237187
time: 0.88 seconds

## Relational analysis of NS_B1_A1_B2_A2_A2

### Relational analysis result of NS_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4266834, upper bound: 339.4253715
time: 1.03 seconds

## BFS NS instance: NS_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -180.1691589, 268.6386108, -132.0075226, 198.9833984, -379.1525574, 400.6461182
1: -141.8278961, 257.1098022, -104.8025665, 191.5839996, -333.4118652, 361.9123230
2: -122.5468063, 265.2631531, -90.4448700, 196.9623718, -319.5091553, 355.7080078
3: -185.2196198, 261.6675415, -136.8857422, 194.5722961, -379.7918396, 398.5532837
4: -147.8655090, 280.6687927, -108.8758087, 208.4512329, -356.3167419, 389.5445557

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_B1_A1_B1

### Relational analysis result of NS_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4236125, upper bound: 339.4283881
time: 0.86 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2

### Relational analysis result of NS_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4236125, upper bound: 339.4318411
time: 0.86 seconds

## BFS NS instance: NS_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -200.1334076, 297.7181091, -132.0075226, 198.9833984, -399.1168213, 429.7256470
1: -157.2297211, 284.7256775, -104.8025665, 191.5839996, -348.8137207, 389.5281677
2: -135.9115906, 293.9135132, -90.4448700, 196.9623718, -332.8739624, 384.3583374
3: -205.4288025, 289.8867188, -136.8857422, 194.5722961, -400.0010986, 426.7724609
4: -164.0981445, 310.9541931, -108.8758087, 208.4512329, -372.5493774, 419.8299561

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4236125, upper bound: 339.4315043
time: 0.88 seconds

## Relational analysis of NS_B1_A2_B1_A2_B2

### Relational analysis result of NS_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4236125, upper bound: 339.4327047
time: 0.98 seconds

## BFS NS instance: NS_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -180.1691589, 268.6386108, -180.1691589, 268.6386108, -448.8077698, 448.8077698
1: -141.8278961, 257.1098022, -141.8278961, 257.1098022, -398.9376831, 398.9376831
2: -122.5468063, 265.2631531, -122.5468063, 265.2631531, -387.8099365, 387.8099365
3: -185.2196198, 261.6675415, -185.2196198, 261.6675415, -446.8871155, 446.8871155
4: -147.8655090, 280.6687927, -147.8655090, 280.6687927, -428.5342712, 428.5342712

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_A2_B2_A1_B1

### Relational analysis result of NS_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4243468, upper bound: 339.4242303
time: 0.90 seconds

## Relational analysis of NS_B1_A2_B2_A1_B2

### Relational analysis result of NS_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4231362, upper bound: 339.4231362
time: 0.94 seconds

## BFS NS instance: NS_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -200.1334076, 297.7181091, -180.1691589, 268.6386108, -468.7720337, 477.8872681
1: -157.2297211, 284.7256775, -141.8278961, 257.1098022, -414.3395386, 426.5535889
2: -135.9115906, 293.9135132, -122.5468063, 265.2631531, -401.1747437, 416.4602966
3: -205.4288025, 289.8867188, -185.2196198, 261.6675415, -467.0963440, 475.1062927
4: -164.0981445, 310.9541931, -147.8655090, 280.6687927, -444.7669067, 458.8196411

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_A2_B2_A2_A1

### Relational analysis result of NS_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4242303, upper bound: 339.4246269
time: 0.64 seconds

## Relational analysis of NS_B1_A2_B2_A2_A2

### Relational analysis result of NS_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4231362, upper bound: 339.4246538
time: 0.81 seconds

## BFS NS instance: NS_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -132.0075226, 198.9833984, -141.9029388, 212.8578644, -344.8653564, 340.8863220
1: -104.8025665, 191.5839996, -112.0533905, 204.8308868, -309.6334229, 303.6373901
2: -90.4448700, 196.9623718, -96.8430252, 210.8139038, -301.2587280, 293.8053894
3: -136.8857422, 194.5722961, -146.4658966, 207.9918671, -344.8775330, 341.0381470
4: -108.8758087, 208.4512329, -116.6867065, 223.1286774, -332.0044861, 325.1379395

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_A1_B1_A1_A1

### Relational analysis result of NS_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4164033, upper bound: 339.4174798
time: 0.95 seconds

## Relational analysis of NS_B2_A1_B1_A1_A2

### Relational analysis result of NS_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4169693, upper bound: 339.4169693
time: 0.66 seconds

## BFS NS instance: NS_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -141.8882599, 212.8376312, -141.9029388, 212.8578644, -354.7461243, 354.7405396
1: -112.0422058, 204.8112488, -112.0533905, 204.8308868, -316.8730774, 316.8646240
2: -96.8333588, 210.7936554, -96.8430252, 210.8139038, -307.6472168, 307.6366577
3: -146.4514313, 207.9716187, -146.4658966, 207.9918671, -354.4432373, 354.4375000
4: -116.6749344, 223.1072540, -116.6867065, 223.1286774, -339.8036194, 339.7939453

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_A1_B1_A2_B1

### Relational analysis result of NS_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4174798, upper bound: 339.4192857
time: 0.95 seconds

## Relational analysis of NS_B2_A1_B1_A2_B2

### Relational analysis result of NS_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4169693, upper bound: 339.4198518
time: 0.92 seconds

## BFS NS instance: NS_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -132.0075226, 198.9833984, -200.1334076, 297.7181091, -429.7256165, 399.1168213
1: -104.8025665, 191.5839996, -157.2297211, 284.7256775, -389.5281677, 348.8137207
2: -90.4448700, 196.9623718, -135.9115906, 293.9135132, -384.3583374, 332.8739624
3: -136.8857422, 194.5722961, -205.4288025, 289.8867188, -426.7724609, 400.0010986
4: -108.8758087, 208.4512329, -164.0981445, 310.9541931, -419.8299561, 372.5493774

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_A1_B2_A1_A1

### Relational analysis result of NS_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4260399, upper bound: 339.4236125
time: 0.84 seconds

## Relational analysis of NS_B2_A1_B2_A1_A2

### Relational analysis result of NS_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4266834, upper bound: 339.4251693
time: 1.17 seconds

## BFS NS instance: NS_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -141.8882599, 212.8376312, -200.1334076, 297.7181091, -439.6063843, 412.9710388
1: -112.0422058, 204.8112488, -157.2297211, 284.7256775, -396.7678833, 362.0409546
2: -96.8333588, 210.7936554, -135.9115906, 293.9135132, -390.7468262, 346.7052307
3: -146.4514313, 207.9716187, -205.4288025, 289.8867188, -436.3381348, 413.4004211
4: -116.6749344, 223.1072540, -164.0981445, 310.9541931, -427.6291199, 387.2053833

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_A1_B2_A2_A1

### Relational analysis result of NS_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4260399, upper bound: 339.4237187
time: 0.90 seconds

## Relational analysis of NS_B2_A1_B2_A2_A2

### Relational analysis result of NS_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4266834, upper bound: 339.4253715
time: 0.94 seconds

## BFS NS instance: NS_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -180.1691589, 268.6386108, -200.1334076, 297.7181091, -477.8872681, 468.7720337
1: -141.8278961, 257.1098022, -157.2297211, 284.7256775, -426.5535583, 414.3395386
2: -122.5468063, 265.2631531, -135.9115906, 293.9135132, -416.4602966, 401.1747437
3: -185.2196198, 261.6675415, -205.4288025, 289.8867188, -475.1062927, 467.0963440
4: -147.8655090, 280.6687927, -164.0981445, 310.9541931, -458.8196411, 444.7669067

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_A2_B2_A1_B1

### Relational analysis result of NS_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4243468, upper bound: 339.4242303
time: 0.87 seconds

## Relational analysis of NS_B2_A2_B2_A1_B2

### Relational analysis result of NS_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4231362, upper bound: 339.4237035
time: 0.93 seconds

## BFS NS instance: NS_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -200.1334076, 297.7181091, -200.1334076, 297.7181091, -497.8515015, 497.8515015
1: -157.2297211, 284.7256775, -157.2297211, 284.7256775, -441.9553833, 441.9553833
2: -135.9115906, 293.9135132, -135.9115906, 293.9135132, -429.8251038, 429.8251038
3: -205.4288025, 289.8867188, -205.4288025, 289.8867188, -495.3155212, 495.3155212
4: -164.0981445, 310.9541931, -164.0981445, 310.9541931, -475.0522766, 475.0522766

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A2_B2_A2_A1

### Relational analysis result of NS_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4242303, upper bound: 339.4251261
time: 0.67 seconds

## Relational analysis of NS_B2_A2_B2_A2_A2

### Relational analysis result of NS_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4231362, upper bound: 339.4252157
time: 0.73 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 7.01 seconds
NS_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 7.01
Output dim: 4, lower bound: -339.4164033, upper bound: 339.4178401
NS_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 7.01
Output dim: 4, lower bound: -339.4169693, upper bound: 339.4182115
NS_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 7.01
Output dim: 4, lower bound: -339.4174798, upper bound: 339.4251716
NS_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 7.01
Output dim: 4, lower bound: -339.4169693, upper bound: 339.4206701
NS_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 7.01
Output dim: 4, lower bound: -339.4260399, upper bound: 339.4236125
NS_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 7.01
Output dim: 4, lower bound: -339.4266834, upper bound: 339.4251693
NS_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 7.01
Output dim: 4, lower bound: -339.4260399, upper bound: 339.4237187
NS_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 7.01
Output dim: 4, lower bound: -339.4266834, upper bound: 339.4253715
NS_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 7.01
Output dim: 4, lower bound: -339.4236125, upper bound: 339.4283881
NS_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 7.01
Output dim: 4, lower bound: -339.4236125, upper bound: 339.4318411
NS_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 7.01
Output dim: 4, lower bound: -339.4236125, upper bound: 339.4315043
NS_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 7.01
Output dim: 4, lower bound: -339.4236125, upper bound: 339.4327047
NS_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 7.01
Output dim: 4, lower bound: -339.4243468, upper bound: 339.4242303
NS_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 7.01
Output dim: 4, lower bound: -339.4231362, upper bound: 339.4231362
NS_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 7.01
Output dim: 4, lower bound: -339.4242303, upper bound: 339.4246269
NS_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 7.01
Output dim: 4, lower bound: -339.4231362, upper bound: 339.4246538
NS_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 7.01
Output dim: 4, lower bound: -339.4164033, upper bound: 339.4174798
NS_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 7.01
Output dim: 4, lower bound: -339.4169693, upper bound: 339.4169693
NS_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 7.01
Output dim: 4, lower bound: -339.4174798, upper bound: 339.4192857
NS_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 7.01
Output dim: 4, lower bound: -339.4169693, upper bound: 339.4198518
NS_B2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 7.01
Output dim: 4, lower bound: -339.4260399, upper bound: 339.4236125
NS_B2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 7.01
Output dim: 4, lower bound: -339.4266834, upper bound: 339.4251693
NS_B2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 7.01
Output dim: 4, lower bound: -339.4260399, upper bound: 339.4237187
NS_B2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 7.01
Output dim: 4, lower bound: -339.4266834, upper bound: 339.4253715
NS_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 7.01
Output dim: 4, lower bound: -339.4243468, upper bound: 339.4242303
NS_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 7.01
Output dim: 4, lower bound: -339.4231362, upper bound: 339.4237035
NS_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 7.01
Output dim: 4, lower bound: -339.4242303, upper bound: 339.4251261
NS_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 7.01
Output dim: 4, lower bound: -339.4231362, upper bound: 339.4252157

## BFS NS instance: NS_B1_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -84.8147354, 128.6466370, -131.2029572, 197.7751770, -282.5899048, 259.8495789
1: -67.6151733, 124.5813828, -104.1700058, 190.4421844, -258.0573120, 228.7513580
2: -58.4463959, 128.0579529, -89.9010773, 195.7871704, -254.2335663, 217.9590302
3: -88.3538437, 125.8761673, -136.0664215, 193.4028778, -281.7566528, 261.9425659
4: -70.3067856, 135.6326599, -108.2207184, 207.2148590, -277.5216064, 243.8533783

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_A1_B1_A1_A1_B1

### Relational analysis result of NS_B1_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4165103, upper bound: 339.4165103
time: 0.64 seconds

## Relational analysis of NS_B1_A1_B1_A1_A1_B2

### Relational analysis result of NS_B1_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4165103, upper bound: 339.4165103
time: 1.03 seconds

## BFS NS instance: NS_B1_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -176.4945679, 266.0788269, -129.2155304, 194.6678162, -371.1623840, 395.2943420
1: -140.1070404, 257.6560364, -102.7059021, 187.6423492, -327.7493896, 360.3619385
2: -121.1961670, 264.3083801, -88.6305161, 192.8322754, -314.0284424, 352.9389038
3: -182.6568146, 260.6510925, -134.1561737, 190.4935150, -373.1503296, 394.8072510
4: -145.7884064, 280.0209351, -106.6847153, 204.1547852, -349.9431458, 386.7056580

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_A1_B1_A1_A2_B1

### Relational analysis result of NS_B1_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4165103, upper bound: 339.4182116
time: 0.90 seconds

## Relational analysis of NS_B1_A1_B1_A1_A2_B2

### Relational analysis result of NS_B1_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4165103, upper bound: 339.4182116
time: 0.91 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -141.1878815, 211.7906036, -84.8147354, 128.6466370, -269.8345337, 296.6053467
1: -111.4851303, 203.8128815, -67.6151733, 124.5813828, -236.0665131, 271.4280396
2: -96.3555527, 209.7696075, -58.4463959, 128.0579529, -224.4134827, 268.2159424
3: -145.7255249, 206.9509125, -88.3538437, 125.8761673, -271.6016846, 295.3046570
4: -116.1019592, 222.0242157, -70.3067856, 135.6326599, -251.7346191, 292.3309937

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4147259, upper bound: 339.4173428
time: 0.94 seconds

## Relational analysis of NS_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4147259, upper bound: 339.4206701
time: 0.92 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -136.7559509, 205.2586212, -176.4945679, 266.0788269, -402.8347778, 381.7531738
1: -108.2838974, 197.8444824, -140.1070404, 257.6560364, -365.9399109, 337.9515076
2: -93.5474472, 203.5740662, -121.1961670, 264.3083801, -357.8558350, 324.7702332
3: -141.6396484, 200.8159180, -182.6568146, 260.6510925, -402.2907410, 383.4727173
4: -112.6449966, 215.6045685, -145.7884064, 280.0209351, -392.6659241, 361.3929443

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4163130, upper bound: 339.4173428
time: 0.89 seconds

## Relational analysis of NS_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4163130, upper bound: 339.4206701
time: 0.72 seconds

## BFS NS instance: NS_B1_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -121.2031631, 182.5256195, -178.0771484, 265.4627075, -386.6658630, 360.6026917
1: -96.2680206, 175.7411346, -140.1763611, 254.0533600, -350.3213806, 315.9174805
2: -83.0352325, 180.7019958, -121.1195526, 262.1324158, -345.1676636, 301.8215332
3: -125.8003311, 178.5331573, -183.0811462, 258.5729065, -384.3731689, 361.6142883
4: -99.9729233, 191.2926941, -146.1492615, 277.3650513, -377.3379517, 337.4419556

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B1_A1_B2_A1_A1_A1

### Relational analysis result of NS_B1_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4246616, upper bound: 339.4206258
time: 0.88 seconds

## Relational analysis of NS_B1_A1_B2_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B1_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_B1_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -135.0331573, 202.9118500, -178.4893494, 266.0621033, -401.0952759, 381.4010620
1: -106.9751587, 194.9790344, -140.4832001, 254.6189423, -361.5940857, 335.4622192
2: -92.2772522, 200.5935364, -121.3839340, 262.7193298, -354.9965515, 321.9774475
3: -139.9173431, 198.1777802, -183.4871979, 259.1484985, -399.0658264, 381.6648865
4: -111.1874847, 212.3526459, -146.4714050, 277.9851990, -389.1726685, 358.8240356

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

## BFS NS instance: NS_B1_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -131.0926666, 196.4400330, -178.0771484, 265.4627075, -396.5553589, 374.5171814
1: -103.5670242, 189.1084595, -140.1763611, 254.0533600, -357.6203308, 329.2848206
2: -89.4489365, 194.6710510, -121.1195526, 262.1324158, -351.5813293, 315.7905884
3: -135.4604492, 192.0638123, -183.0811462, 258.5729065, -394.0333252, 375.1449280
4: -107.7420044, 206.1067505, -146.1492615, 277.3650513, -385.1070557, 352.2560120

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B1_A1_B2_A2_A1_A1

### Relational analysis result of NS_B1_A1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4223506, upper bound: 339.4209035
time: 0.96 seconds

## Relational analysis of NS_B1_A1_B2_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B1_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_B1_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -143.4188995, 214.6302185, -178.4893494, 266.0621033, -409.4810181, 393.1194763
1: -113.1838760, 206.2543030, -140.4832001, 254.6189423, -367.8028259, 346.7374573
2: -97.7532501, 212.4093628, -121.3839340, 262.7193298, -360.4725342, 333.7933044
3: -148.1632385, 209.6267853, -183.4871979, 259.1484985, -407.3117371, 393.1139526
4: -117.8254166, 224.9079895, -146.4714050, 277.9851990, -395.8106079, 371.3793945

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_A1_B2_A2_A2_A1

### Relational analysis result of NS_B1_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4254607, upper bound: 339.4245655
time: 1.01 seconds

## Relational analysis of NS_B1_A1_B2_A2_A2_A2

### Relational analysis result of NS_B1_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4254727, upper bound: 339.4245925
time: 1.11 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -178.0771484, 265.4627075, -121.2031631, 182.5256195, -360.6026917, 386.6658630
1: -140.1763611, 254.0533600, -96.2680206, 175.7411346, -315.9174805, 350.3213806
2: -121.1195526, 262.1324158, -83.0352325, 180.7019958, -301.8215332, 345.1676636
3: -183.0811462, 258.5729065, -125.8003311, 178.5331573, -361.6143188, 384.3731689
4: -146.1492615, 277.3650513, -99.9729233, 191.2926941, -337.4419556, 377.3379517

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B1_A2_B1_A1_B1_B1

### Relational analysis result of NS_B1_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4206258, upper bound: 339.4246616
time: 0.91 seconds

## Relational analysis of NS_B1_A2_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 35

## BFS NS instance: NS_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -178.4893494, 266.0621033, -135.0331573, 202.9118500, -381.4010620, 401.0952759
1: -140.4832001, 254.6189423, -106.9751587, 194.9790344, -335.4622192, 361.5940857
2: -121.3839340, 262.7193298, -92.2772522, 200.5935364, -321.9774475, 354.9965515
3: -183.4871979, 259.1484985, -139.9173431, 198.1777802, -381.6648865, 399.0658264
4: -146.4714050, 277.9851990, -111.1874847, 212.3526459, -358.8240356, 389.1726685

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_A2_B1_A1_B2_B1

### Relational analysis result of NS_B1_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4244033, upper bound: 339.4306280
time: 0.90 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2_B2

### Relational analysis result of NS_B1_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4242693, upper bound: 339.4300457
time: 0.99 seconds

## BFS NS instance: NS_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -198.2290649, 294.8228760, -121.2031631, 182.5256195, -380.7546387, 416.0260315
1: -155.7336426, 281.9318237, -96.2680206, 175.7411346, -331.4747925, 378.1998291
2: -134.6158752, 291.0564270, -83.0352325, 180.7019958, -315.3178711, 374.0916443
3: -203.4941406, 287.0607605, -125.8003311, 178.5331573, -382.0272827, 412.8610840
4: -162.5386810, 307.9455872, -99.9729233, 191.2926941, -353.8313599, 407.9184875

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B1_A2_B1_A2_B1_B1

### Relational analysis result of NS_B1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4205892, upper bound: 339.4258521
time: 0.85 seconds

## Relational analysis of NS_B1_A2_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4173759, upper bound: 339.4208035
time: 1.46 seconds

## Relational analysis of NS_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_A2_B1_A2_B1_B1

### Relational analysis result of NS_B1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4188100, upper bound: 339.4313783
time: 1.02 seconds

## Relational analysis of NS_B1_A2_B1_A2_B1_B2

### Relational analysis result of NS_B1_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4232782, upper bound: 339.4313220
time: 0.99 seconds

## BFS NS instance: NS_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -198.4437256, 295.1159973, -135.0331573, 202.9118500, -401.3554688, 430.1491699
1: -155.8816223, 282.2169189, -106.9751587, 194.9790344, -350.8605957, 389.1920776
2: -134.7429657, 291.3504333, -92.2772522, 200.5935364, -335.3364868, 383.6276245
3: -203.6920624, 287.3535767, -139.9173431, 198.1777802, -401.8697815, 427.2709045
4: -162.6961517, 308.2540283, -111.1874847, 212.3526459, -375.0487366, 419.4415283

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4203873, upper bound: 339.4286223
time: 0.67 seconds

## Relational analysis of NS_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4203873, upper bound: 339.4327047
time: 1.06 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -180.1691589, 268.6386108, -177.1547394, 264.3292236, -444.4983826, 445.7932739
1: -141.8278961, 257.1098022, -139.6272736, 253.1960449, -395.0239258, 396.7370605
2: -122.5468063, 265.2631531, -120.6045532, 261.1123657, -383.6591492, 385.8677063
3: -185.2196198, 261.6675415, -182.2870636, 257.7313843, -442.9509888, 443.9544983
4: -147.8655090, 280.6687927, -145.5399628, 276.2418518, -424.1073303, 426.2086792

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4231362, upper bound: 339.4231362
time: 0.61 seconds

## Relational analysis of NS_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4231362, upper bound: 339.4231362
time: 0.73 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -180.1691589, 268.6386108, -177.6839905, 264.9102783, -445.0794373, 446.3226013
1: -141.8278961, 257.1098022, -139.8705139, 253.5063019, -395.3341370, 396.9803162
2: -122.5468063, 265.2631531, -120.8578186, 261.5688171, -384.1156006, 386.1209717
3: -185.2196198, 261.6675415, -182.6897888, 258.0085449, -443.2281494, 444.3572693
4: -147.8655090, 280.6687927, -145.8274994, 276.7703552, -424.6358337, 426.4962769

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4231362, upper bound: 339.4231362
time: 0.85 seconds

## Relational analysis of NS_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4231362, upper bound: 339.4231362
time: 0.84 seconds

## BFS NS instance: NS_B1_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -196.9746552, 293.2734375, -180.1691589, 268.6386108, -465.6132507, 473.4425964
1: -154.8932800, 280.6892090, -141.8278961, 257.1098022, -412.0030823, 422.5170898
2: -133.8690796, 289.6318665, -122.5468063, 265.2631531, -399.1322021, 412.1786194
3: -202.3011475, 285.7830200, -185.2196198, 261.6675415, -463.9686584, 471.0026245
4: -161.6605377, 306.3764954, -147.8655090, 280.6687927, -442.3292542, 454.2419739

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B1_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_A2_B2_A2_A1_B1

### Relational analysis result of NS_B1_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4237035, upper bound: 339.4246269
time: 1.08 seconds

## Relational analysis of NS_B1_A2_B2_A2_A1_B2

### Relational analysis result of NS_B1_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4237035, upper bound: 339.4246269
time: 1.04 seconds

## BFS NS instance: NS_B1_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -197.6980591, 294.0541992, -180.1691589, 268.6386108, -466.3366699, 474.2233276
1: -155.3034973, 281.1775818, -141.8278961, 257.1098022, -412.4132996, 423.0054321
2: -134.2444611, 290.2764587, -122.5468063, 265.2631531, -399.5076294, 412.8232117
3: -202.9371185, 286.2823792, -185.2196198, 261.6675415, -464.6046448, 471.5019531
4: -162.0886688, 307.1148987, -147.8655090, 280.6687927, -442.7574463, 454.9803467

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_A2_B2_A2_A2_B1

### Relational analysis result of NS_B1_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4237035, upper bound: 339.4246538
time: 1.01 seconds

## Relational analysis of NS_B1_A2_B2_A2_A2_B2

### Relational analysis result of NS_B1_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4237035, upper bound: 339.4246538
time: 1.03 seconds

## BFS NS instance: NS_B2_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -84.8147354, 128.6466370, -141.2022095, 211.8103333, -296.6250610, 269.8488464
1: -67.6151733, 124.5813828, -111.4960403, 203.8319702, -271.4471436, 236.0774231
2: -58.4463959, 128.0579529, -96.3649750, 209.7893219, -268.2357178, 224.4229126
3: -88.3538437, 125.8761673, -145.7396393, 206.9706726, -295.3243713, 271.6157837
4: -70.3067856, 135.6326599, -116.1134033, 222.0451355, -292.3518982, 251.7460632

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_A1_B1_A1_A1_B1

### Relational analysis result of NS_B2_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4165103, upper bound: 339.4147259
time: 0.70 seconds

## Relational analysis of NS_B2_A1_B1_A1_A1_B2

### Relational analysis result of NS_B2_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4173428, upper bound: 339.4153822
time: 0.89 seconds

## BFS NS instance: NS_B2_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -176.4945679, 266.0788269, -136.7559509, 205.2586212, -381.7531738, 402.8347778
1: -140.1070404, 257.6560364, -108.2838974, 197.8444824, -337.9515076, 365.9399109
2: -121.1961670, 264.3083801, -93.5474472, 203.5740662, -324.7702332, 357.8558350
3: -182.6568146, 260.6510925, -141.6396484, 200.8159180, -383.4727173, 402.2907410
4: -145.7884064, 280.0209351, -112.6449966, 215.6045685, -361.3929443, 392.6659241

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_A1_B1_A1_A2_B1

### Relational analysis result of NS_B2_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4173428, upper bound: 339.4163130
time: 0.94 seconds

## Relational analysis of NS_B2_A1_B1_A1_A2_B2

### Relational analysis result of NS_B2_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4165103, upper bound: 339.4169693
time: 0.72 seconds

## BFS NS instance: NS_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -141.1878815, 211.7906036, -98.8425827, 148.5214691, -289.7093506, 310.6331787
1: -111.4851303, 203.8128815, -78.1776886, 143.6714935, -255.1565857, 281.9905701
2: -96.3555527, 209.7696075, -67.6280899, 147.9380035, -244.2935333, 277.3976746
3: -145.7255249, 206.9509125, -102.3993759, 145.3697205, -291.0952454, 309.3502502
4: -116.1019592, 222.0242157, -81.4340668, 156.7362823, -272.8382263, 303.4582825

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4147259, upper bound: 339.4161666
time: 0.82 seconds

## Relational analysis of NS_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4147259, upper bound: 339.4191955
time: 0.93 seconds

## BFS NS instance: NS_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -136.7559509, 205.2586212, -184.0684052, 276.5903320, -413.3462524, 389.3270264
1: -108.2838974, 197.8444824, -145.6859741, 267.2598267, -375.5437012, 343.5303345
2: -93.5474472, 203.5740662, -126.0879517, 274.5840149, -368.1314697, 329.6620178
3: -141.6396484, 200.8159180, -190.1156464, 270.6260376, -412.2656860, 390.9315796
4: -112.6449966, 215.6045685, -151.7548981, 290.9830627, -403.6280518, 367.3594666

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4163130, upper bound: 339.4168227
time: 0.86 seconds

## Relational analysis of NS_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4163130, upper bound: 339.4198518
time: 1.02 seconds

## BFS NS instance: NS_B2_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -121.2031631, 182.5256195, -198.2290649, 294.8228760, -416.0260315, 380.7546387
1: -96.2680206, 175.7411346, -155.7336426, 281.9318237, -378.1998291, 331.4747925
2: -83.0352325, 180.7019958, -134.6158752, 291.0564270, -374.0916443, 315.3178711
3: -125.8003311, 178.5331573, -203.4941406, 287.0607605, -412.8610840, 382.0272827
4: -99.9729233, 191.2926941, -162.5386810, 307.9455872, -407.9184875, 353.8313599

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_A1_B2_A1_A1_A1

### Relational analysis result of NS_B2_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4258521, upper bound: 339.4205892
time: 1.08 seconds

## Relational analysis of NS_B2_A1_B2_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A1_B2_A1_A1_A1

### Relational analysis result of NS_B2_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4313783, upper bound: 339.4188100
time: 0.95 seconds

## Relational analysis of NS_B2_A1_B2_A1_A1_A2

### Relational analysis result of NS_B2_A1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4271684, upper bound: 339.4232782
time: 1.05 seconds

## BFS NS instance: NS_B2_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -135.0331573, 202.9118500, -198.4437256, 295.1159973, -430.1491699, 401.3554993
1: -106.9751587, 194.9790344, -155.8816223, 282.2169189, -389.1920776, 350.8605957
2: -92.2772522, 200.5935364, -134.7429657, 291.3504333, -383.6276245, 335.3364868
3: -139.9173431, 198.1777802, -203.6920624, 287.3535767, -427.2709045, 401.8697815
4: -111.1874847, 212.3526459, -162.6961517, 308.2540283, -419.4415283, 375.0487366

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A1_B2_A1_A2_B1

### Relational analysis result of NS_B2_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4286223, upper bound: 339.4203873
time: 1.06 seconds

## Relational analysis of NS_B2_A1_B2_A1_A2_B2

### Relational analysis result of NS_B2_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4286223, upper bound: 339.4251693
time: 0.76 seconds

## BFS NS instance: NS_B2_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -131.0926666, 196.4400330, -198.2290649, 294.8228760, -425.9155273, 394.6690979
1: -103.5670242, 189.1084595, -155.7336426, 281.9318237, -385.4988098, 344.8421021
2: -89.4489365, 194.6710510, -134.6158752, 291.0564270, -380.5053101, 329.2869263
3: -135.4604492, 192.0638123, -203.4941406, 287.0607605, -422.5212097, 395.5579529
4: -107.7420044, 206.1067505, -162.5386810, 307.9455872, -415.6875916, 368.6454163

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_A1_B2_A2_A1_A1

### Relational analysis result of NS_B2_A1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4223506, upper bound: 339.4208889
time: 1.04 seconds

## Relational analysis of NS_B2_A1_B2_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_B2_A2_A1_B1

### Relational analysis result of NS_B2_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4195653, upper bound: 339.4183911
time: 1.02 seconds

## Relational analysis of NS_B2_A1_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A1_B2_A2_A1_B1

### Relational analysis result of NS_B2_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4192314, upper bound: 339.4179816
time: 0.97 seconds

## Relational analysis of NS_B2_A1_B2_A2_A1_B2

### Relational analysis result of NS_B2_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4192314, upper bound: 339.4237187
time: 0.95 seconds

## BFS NS instance: NS_B2_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -143.4188995, 214.6302185, -198.4437256, 295.1159973, -438.5349121, 413.0738831
1: -113.1838760, 206.2543030, -155.8816223, 282.2169189, -395.4007874, 362.1358643
2: -97.7532501, 212.4093628, -134.7429657, 291.3504333, -389.1036072, 347.1523438
3: -148.1632385, 209.6267853, -203.6920624, 287.3535767, -435.5168152, 413.3188171
4: -117.8254166, 224.9079895, -162.6961517, 308.2540283, -426.0794373, 387.6040955

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A1_B2_A2_A2_B1

### Relational analysis result of NS_B2_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4195393, upper bound: 339.4179817
time: 1.01 seconds

## Relational analysis of NS_B2_A1_B2_A2_A2_B2

### Relational analysis result of NS_B2_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4195392, upper bound: 339.4253715
time: 0.88 seconds

## BFS NS instance: NS_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -180.1691589, 268.6386108, -196.9746552, 293.2734375, -473.4425964, 465.6132507
1: -141.8278961, 257.1098022, -154.8932800, 280.6892090, -422.5170898, 412.0030823
2: -122.5468063, 265.2631531, -133.8690796, 289.6318665, -412.1786194, 399.1322021
3: -185.2196198, 261.6675415, -202.3011475, 285.7830200, -471.0026245, 463.9686279
4: -147.8655090, 280.6687927, -161.6605377, 306.3764954, -454.2419739, 442.3292542

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4231362, upper bound: 339.4237035
time: 0.98 seconds

## Relational analysis of NS_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4246269, upper bound: 339.4237035
time: 0.96 seconds

## BFS NS instance: NS_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -180.1691589, 268.6386108, -197.6980591, 294.0541992, -474.2233276, 466.3366699
1: -141.8278961, 257.1098022, -155.3034973, 281.1775818, -423.0054626, 412.4132996
2: -122.5468063, 265.2631531, -134.2444611, 290.2764587, -412.8232117, 399.5076294
3: -185.2196198, 261.6675415, -202.9371185, 286.2823792, -471.5019531, 464.6046448
4: -147.8655090, 280.6687927, -162.0886688, 307.1148987, -454.9803467, 442.7574463

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4246538, upper bound: 339.4237035
time: 0.92 seconds

## Relational analysis of NS_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4246538, upper bound: 339.4237035
time: 0.85 seconds

## BFS NS instance: NS_B2_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -196.9746552, 293.2734375, -200.1334076, 297.7181091, -494.6927490, 493.4068604
1: -154.8932800, 280.6892090, -157.2297211, 284.7256775, -439.6189575, 437.9189453
2: -133.8690796, 289.6318665, -135.9115906, 293.9135132, -427.7825317, 425.5434570
3: -202.3011475, 285.7830200, -205.4288025, 289.8867188, -492.1878357, 491.2118225
4: -161.6605377, 306.3764954, -164.0981445, 310.9541931, -472.6146240, 470.4746094

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B2_A2_B2_A2_A1_B1

### Relational analysis result of NS_B2_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4202622, upper bound: 339.4198564
time: 0.93 seconds

## Relational analysis of NS_B2_A2_B2_A2_A1_B2

### Relational analysis result of NS_B2_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4209729, upper bound: 339.4207809
time: 0.70 seconds

## BFS NS instance: NS_B2_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -197.6980591, 294.0541992, -200.1334076, 297.7181091, -495.4161682, 494.1876221
1: -155.3034973, 281.1775818, -157.2297211, 284.7256775, -440.0291748, 438.4072876
2: -134.2444611, 290.2764587, -135.9115906, 293.9135132, -428.1579285, 426.1880493
3: -202.9371185, 286.2823792, -205.4288025, 289.8867188, -492.8238220, 491.7111816
4: -162.0886688, 307.1148987, -164.0981445, 310.9541931, -473.0428467, 471.2129822

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B2_A2_B2_A2_A2_B1

### Relational analysis result of NS_B2_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4202290, upper bound: 339.4197757
time: 0.83 seconds

## Relational analysis of NS_B2_A2_B2_A2_A2_B2

### Relational analysis result of NS_B2_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4205730, upper bound: 339.4205466
time: 1.05 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.48 seconds
NS_B1_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4165103, upper bound: 339.4165103
NS_B1_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4165103, upper bound: 339.4165103
NS_B1_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4165103, upper bound: 339.4182116
NS_B1_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4165103, upper bound: 339.4182116
NS_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4147259, upper bound: 339.4173428
NS_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4147259, upper bound: 339.4206701
NS_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4163130, upper bound: 339.4173428
NS_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4163130, upper bound: 339.4206701
NS_B1_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4254607, upper bound: 339.4245655
NS_B1_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4254727, upper bound: 339.4245925
NS_B1_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4244033, upper bound: 339.4306280
NS_B1_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4242693, upper bound: 339.4300457
NS_B1_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4188100, upper bound: 339.4313783
NS_B1_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4232782, upper bound: 339.4313220
NS_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4203873, upper bound: 339.4286223
NS_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4203873, upper bound: 339.4327047
NS_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4231362, upper bound: 339.4231362
NS_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4231362, upper bound: 339.4231362
NS_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4231362, upper bound: 339.4231362
NS_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4231362, upper bound: 339.4231362
NS_B1_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4237035, upper bound: 339.4246269
NS_B1_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4237035, upper bound: 339.4246269
NS_B1_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4237035, upper bound: 339.4246538
NS_B1_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4237035, upper bound: 339.4246538
NS_B2_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4165103, upper bound: 339.4147259
NS_B2_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4173428, upper bound: 339.4153822
NS_B2_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4173428, upper bound: 339.4163130
NS_B2_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4165103, upper bound: 339.4169693
NS_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4147259, upper bound: 339.4161666
NS_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4147259, upper bound: 339.4191955
NS_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4163130, upper bound: 339.4168227
NS_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4163130, upper bound: 339.4198518
NS_B2_A1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4313783, upper bound: 339.4188100
NS_B2_A1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4271684, upper bound: 339.4232782
NS_B2_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4286223, upper bound: 339.4203873
NS_B2_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4286223, upper bound: 339.4251693
NS_B2_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4192314, upper bound: 339.4179816
NS_B2_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4192314, upper bound: 339.4237187
NS_B2_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4195393, upper bound: 339.4179817
NS_B2_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4195392, upper bound: 339.4253715
NS_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4231362, upper bound: 339.4237035
NS_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4246269, upper bound: 339.4237035
NS_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4246538, upper bound: 339.4237035
NS_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4246538, upper bound: 339.4237035
NS_B2_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4202622, upper bound: 339.4198564
NS_B2_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4209729, upper bound: 339.4207809
NS_B2_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4202290, upper bound: 339.4197757
NS_B2_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 4, lower bound: -339.4205730, upper bound: 339.4205466

## BFS NS instance: NS_B1_A1_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -84.8147354, 128.6466370, -84.8147354, 128.6466370, -213.4613647, 213.4613647
1: -67.6151733, 124.5813828, -67.6151733, 124.5813828, -192.1965637, 192.1965637
2: -58.4463959, 128.0579529, -58.4463959, 128.0579529, -186.5043488, 186.5043488
3: -88.3538437, 125.8761673, -88.3538437, 125.8761673, -214.2299957, 214.2299957
4: -70.3067856, 135.6326599, -70.3067856, 135.6326599, -205.9394226, 205.9394226

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A1_B1_A1_A1_B1_A1

### Relational analysis result of NS_B1_A1_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4199883, upper bound: 339.4138910
time: 0.91 seconds

## Relational analysis of NS_B1_A1_B1_A1_A1_B1_A2

### Relational analysis result of NS_B1_A1_B1_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4156850, upper bound: 339.4133167
time: 0.62 seconds

## BFS NS instance: NS_B1_A1_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -84.8147354, 128.6466370, -176.4945679, 266.0788269, -350.8935547, 305.1412048
1: -67.6151733, 124.5813828, -140.1070404, 257.6560364, -325.2712097, 264.6884155
2: -58.4463959, 128.0579529, -121.1961670, 264.3083801, -322.7547302, 249.2540588
3: -88.3538437, 125.8761673, -182.6568146, 260.6510925, -349.0048828, 308.5329895
4: -70.3067856, 135.6326599, -145.7884064, 280.0209351, -350.3277283, 281.4210815

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 9

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A1_B1_A1_A1_B2_A1

### Relational analysis result of NS_B1_A1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4229320, upper bound: 339.4116518
time: 1.03 seconds

## Relational analysis of NS_B1_A1_B1_A1_A1_B2_A2

### Relational analysis result of NS_B1_A1_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4231019, upper bound: 339.4178401
time: 0.96 seconds

## BFS NS instance: NS_B1_A1_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -176.4945679, 266.0788269, -84.8147354, 128.6466370, -305.1412048, 350.8935547
1: -140.1070404, 257.6560364, -67.6151733, 124.5813828, -264.6884155, 325.2712097
2: -121.1961670, 264.3083801, -58.4463959, 128.0579529, -249.2540588, 322.7547302
3: -182.6568146, 260.6510925, -88.3538437, 125.8761673, -308.5329895, 349.0049133
4: -145.7884064, 280.0209351, -70.3067856, 135.6326599, -281.4210815, 350.3277283

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 9

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1_A1_B1_A1_A2_B1_B1

### Relational analysis result of NS_B1_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4116518, upper bound: 339.4181886
time: 0.99 seconds

## Relational analysis of NS_B1_A1_B1_A1_A2_B1_B2

### Relational analysis result of NS_B1_A1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4164740, upper bound: 339.4180384
time: 1.00 seconds

## BFS NS instance: NS_B1_A1_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -176.4945679, 266.0788269, -176.4945679, 266.0788269, -442.5733948, 442.5733948
1: -140.1070404, 257.6560364, -140.1070404, 257.6560364, -397.7630615, 397.7630615
2: -121.1961670, 264.3083801, -121.1961670, 264.3083801, -385.5045471, 385.5045471
3: -182.6568146, 260.6510925, -182.6568146, 260.6510925, -443.3079224, 443.3079224
4: -145.7884064, 280.0209351, -145.7884064, 280.0209351, -425.8093262, 425.8093262

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_A1_B1_A1_A2_B2_A1

### Relational analysis result of NS_B1_A1_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4162512, upper bound: 339.4171973
time: 0.80 seconds

## Relational analysis of NS_B1_A1_B1_A1_A2_B2_A2

### Relational analysis result of NS_B1_A1_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4156101, upper bound: 339.4171973
time: 0.82 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -98.8425827, 148.5214691, -84.8147354, 128.6466370, -227.4892120, 233.3362122
1: -78.1776886, 143.6714935, -67.6151733, 124.5813828, -202.7590637, 211.2866516
2: -67.6280899, 147.9380035, -58.4463959, 128.0579529, -195.6860199, 206.3843994
3: -102.3993759, 145.3697205, -88.3538437, 125.8761673, -228.2755127, 233.7235413
4: -81.4340668, 156.7362823, -70.3067856, 135.6326599, -217.0667114, 227.0430603

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_B1_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4079511, upper bound: 339.4094795
time: 0.94 seconds

## Relational analysis of NS_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4132048, upper bound: 339.4215167
time: 1.06 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -184.0684052, 276.5903320, -84.8147354, 128.6466370, -312.7150269, 361.4050598
1: -145.6859741, 267.2598267, -67.6151733, 124.5813828, -270.2672729, 334.8750000
2: -126.0879517, 274.5840149, -58.4463959, 128.0579529, -254.1458740, 333.0303345
3: -190.1156464, 270.6260376, -88.3538437, 125.8761673, -315.9917603, 358.9798279
4: -151.7548981, 290.9830627, -70.3067856, 135.6326599, -287.3875732, 361.2898560

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 9

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4099035, upper bound: 339.4242462
time: 0.89 seconds

## Relational analysis of NS_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4168246, upper bound: 339.4251699
time: 1.04 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -98.8425827, 148.5214691, -176.4945679, 266.0788269, -364.9213562, 325.0160522
1: -78.1776886, 143.6714935, -140.1070404, 257.6560364, -335.8337402, 283.7785339
2: -67.6280899, 147.9380035, -121.1961670, 264.3083801, -331.9364624, 269.1341553
3: -102.3993759, 145.3697205, -182.6568146, 260.6510925, -363.0504761, 328.0265503
4: -81.4340668, 156.7362823, -145.7884064, 280.0209351, -361.4549866, 302.5246582

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A1_B1_A2_B2_A1_A1

### Relational analysis result of NS_B1_A1_B1_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4084155, upper bound: 339.4136307
time: 0.79 seconds

## Relational analysis of NS_B1_A1_B1_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4133689, upper bound: 339.4171054
time: 1.02 seconds

## Relational analysis of NS_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4133689, upper bound: 339.4170480
time: 1.08 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -184.0684052, 276.5903320, -176.4945679, 266.0788269, -450.1472168, 453.0848999
1: -145.6859741, 267.2598267, -140.1070404, 257.6560364, -403.3419800, 407.3668823
2: -126.0879517, 274.5840149, -121.1961670, 264.3083801, -390.3963318, 395.7801819
3: -190.1156464, 270.6260376, -182.6568146, 260.6510925, -450.7666626, 453.2828369
4: -151.7548981, 290.9830627, -145.7884064, 280.0209351, -431.7758179, 436.7714539

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4133689, upper bound: 339.4204931
time: 0.92 seconds

## Relational analysis of NS_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4133689, upper bound: 339.4200797
time: 1.05 seconds

## BFS NS instance: NS_B1_A1_B2_A2_A2_A1

### Backsubstitution after applying NS history:
0: -142.3289795, 213.2878113, -178.4893494, 266.0621033, -408.3910828, 391.7770081
1: -112.5761032, 205.1165161, -140.4832001, 254.6189423, -367.1950378, 345.5997314
2: -97.1610794, 211.1718750, -121.3839340, 262.7193298, -359.8804016, 332.5558167
3: -147.2827606, 208.5200958, -183.4871979, 259.1484985, -406.4312134, 392.0072937
4: -117.1281509, 223.5491486, -146.4714050, 277.9851990, -395.1132812, 370.0205383

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1_A1_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B1_A1_B2_A2_A2_A1_A1

### Relational analysis result of NS_B1_A1_B2_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4213997, upper bound: 339.4191128
time: 0.92 seconds

## Relational analysis of NS_B1_A1_B2_A2_A2_A1_A2

### Relational analysis result of NS_B1_A1_B2_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4211617, upper bound: 339.4205840
time: 1.16 seconds

## BFS NS instance: NS_B1_A1_B2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -140.8958435, 210.8361816, -178.4893494, 266.0621033, -406.9579468, 389.3254089
1: -111.2125854, 202.5887451, -140.4832001, 254.6189423, -365.8315125, 343.0718994
2: -96.0423660, 208.6501007, -121.3839340, 262.7193298, -358.7616882, 330.0340271
3: -145.6230469, 205.9089050, -183.4871979, 259.1484985, -404.7715454, 389.3960876
4: -115.7536697, 220.9487610, -146.4714050, 277.9851990, -393.7388611, 367.4201660

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1_A1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B1_A1_B2_A2_A2_A2_A1

### Relational analysis result of NS_B1_A1_B2_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4214701, upper bound: 339.4203280
time: 0.98 seconds

## Relational analysis of NS_B1_A1_B2_A2_A2_A2_A2

### Relational analysis result of NS_B1_A1_B2_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4211303, upper bound: 339.4200758
time: 1.14 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -178.4893494, 266.0621033, -133.5042419, 200.9851379, -379.4743652, 399.5663452
1: -140.4832001, 254.6189423, -105.9809799, 193.1136932, -333.5968628, 360.5999146
2: -121.3839340, 262.7193298, -91.3644867, 198.6595917, -320.0435181, 354.0838013
3: -183.4871979, 259.1484985, -138.5200500, 196.3673706, -379.8545532, 397.6685181
4: -146.4714050, 277.9851990, -110.1298065, 210.2101288, -356.6815186, 388.1149902

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_A2_B1_A1_B2_B1_A1

### Relational analysis result of NS_B1_A2_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4242693, upper bound: 339.4300457
time: 0.93 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2_B1_A2

### Relational analysis result of NS_B1_A2_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4242693, upper bound: 339.4300457
time: 1.04 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -178.4893494, 266.0621033, -132.7802124, 199.5456085, -378.0348816, 398.8423157
1: -140.4832001, 254.6189423, -105.2074051, 191.7133636, -332.1965637, 359.8263550
2: -121.3839340, 262.7193298, -90.7458344, 197.2406311, -318.6245728, 353.4651489
3: -183.4871979, 259.1484985, -137.6270599, 194.8667908, -378.3539124, 396.7755737
4: -146.4714050, 277.9851990, -109.3366623, 208.8111572, -355.2825623, 387.3218689

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A2_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_A2_B1_A1_B2_B2_A1

### Relational analysis result of NS_B1_A2_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4228289, upper bound: 339.4300457
time: 1.00 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2_B2_A2

### Relational analysis result of NS_B1_A2_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4228289, upper bound: 339.4300457
time: 0.99 seconds

## BFS NS instance: NS_B1_A2_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -198.2290649, 294.8228760, -119.7156830, 180.5114136, -378.7404785, 414.5385742
1: -155.7336426, 281.9318237, -95.3078690, 173.8383484, -329.5719604, 377.2396851
2: -134.6158752, 291.0564270, -82.1589203, 178.6991577, -313.3150330, 373.2153320
3: -203.4941406, 287.0607605, -124.4761047, 176.6610718, -380.1551514, 411.5368652
4: -162.5386810, 307.9455872, -98.9478302, 189.1291656, -351.6677551, 406.8933716

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_B1_A2_B1_B1_A1

### Relational analysis result of NS_B1_A2_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4129993, upper bound: 339.4269406
time: 0.81 seconds

## Relational analysis of NS_B1_A2_B1_A2_B1_B1_A2

### Relational analysis result of NS_B1_A2_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4129993, upper bound: 339.4313783
time: 0.96 seconds

## BFS NS instance: NS_B1_A2_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -198.2290649, 294.8228760, -118.9497910, 179.1519012, -377.3809814, 413.7726440
1: -155.7336426, 281.9318237, -94.5011826, 172.4779816, -328.2115479, 376.4329834
2: -134.6158752, 291.0564270, -81.5044937, 177.3490448, -311.9649048, 372.5608826
3: -203.4941406, 287.0607605, -123.5130386, 175.2167664, -378.7108765, 410.5737915
4: -162.5386810, 307.9455872, -98.1219101, 187.7532043, -350.2918396, 406.0675049

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_B1_A2_B1_B2_A1

### Relational analysis result of NS_B1_A2_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4188715, upper bound: 339.4290488
time: 0.96 seconds

## Relational analysis of NS_B1_A2_B1_A2_B1_B2_A2

### Relational analysis result of NS_B1_A2_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4188715, upper bound: 339.4313220
time: 1.01 seconds

## BFS NS instance: NS_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -190.5328827, 283.1501160, -135.0331573, 202.9118500, -393.4447021, 418.1832275
1: -149.6785889, 270.6842041, -106.9751587, 194.9790344, -344.6576233, 377.6593628
2: -129.3718872, 279.5513000, -92.2772522, 200.5935364, -329.9653931, 371.8284607
3: -195.6701355, 275.6775513, -139.9173431, 198.1777802, -393.8479004, 415.5948486
4: -156.2226410, 295.8223877, -111.1874847, 212.3526459, -368.5752869, 407.0098877

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4129993, upper bound: 339.4282826
time: 1.09 seconds

## Relational analysis of NS_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4188715, upper bound: 339.4283144
time: 0.93 seconds

## BFS NS instance: NS_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -201.8305817, 299.9625854, -135.0331573, 202.9118500, -404.7423706, 434.9957275
1: -158.4784241, 286.6781616, -106.9751587, 194.9790344, -353.4574585, 393.6533203
2: -136.9770660, 296.0209961, -92.2772522, 200.5935364, -337.5705566, 388.2981567
3: -207.1752930, 292.0406494, -139.9173431, 198.1777802, -405.3530273, 431.9579773
4: -165.4182587, 313.2349548, -111.1874847, 212.3526459, -377.7709045, 424.4224243

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4129993, upper bound: 339.4269406
time: 0.92 seconds

## Relational analysis of NS_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4188715, upper bound: 339.4323279
time: 1.16 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -177.1547394, 264.3292236, -177.1547394, 264.3292236, -441.4838867, 441.4838867
1: -139.6272736, 253.1960449, -139.6272736, 253.1960449, -392.8233032, 392.8233032
2: -120.6045532, 261.1123657, -120.6045532, 261.1123657, -381.7169189, 381.7169189
3: -182.2870636, 257.7313843, -182.2870636, 257.7313843, -440.0184021, 440.0184021
4: -145.5399628, 276.2418518, -145.5399628, 276.2418518, -421.7817383, 421.7817383

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -177.6839905, 264.9102783, -177.1547394, 264.3292236, -442.0132141, 442.0649414
1: -139.8705139, 253.5063019, -139.6272736, 253.1960449, -393.0665588, 393.1335144
2: -120.8578186, 261.5688171, -120.6045532, 261.1123657, -381.9701843, 382.1733704
3: -182.6897888, 258.0085449, -182.2870636, 257.7313843, -440.4211731, 440.2954712
4: -145.8274994, 276.7703552, -145.5399628, 276.2418518, -422.0693359, 422.3102417

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## BFS NS instance: NS_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -177.1547394, 264.3292236, -177.6839905, 264.9102783, -442.0649414, 442.0132141
1: -139.6272736, 253.1960449, -139.8705139, 253.5063019, -393.1335144, 393.0665588
2: -120.6045532, 261.1123657, -120.8578186, 261.5688171, -382.1733704, 381.9701843
3: -182.2870636, 257.7313843, -182.6897888, 258.0085449, -440.2954712, 440.4211731
4: -145.5399628, 276.2418518, -145.8274994, 276.7703552, -422.3102417, 422.0693359

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## BFS NS instance: NS_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -177.6839905, 264.9102783, -177.6839905, 264.9102783, -442.5942688, 442.5942688
1: -139.8705139, 253.5063019, -139.8705139, 253.5063019, -393.3768005, 393.3768005
2: -120.8578186, 261.5688171, -120.8578186, 261.5688171, -382.4266357, 382.4266357
3: -182.6897888, 258.0085449, -182.6897888, 258.0085449, -440.6982727, 440.6982727
4: -145.8274994, 276.7703552, -145.8274994, 276.7703552, -422.5978394, 422.5978394

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## BFS NS instance: NS_B1_A2_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -196.9746552, 293.2734375, -177.1547394, 264.3292236, -461.3038635, 470.4281006
1: -154.8932800, 280.6892090, -139.6272736, 253.1960449, -408.0893250, 420.3164673
2: -133.8690796, 289.6318665, -120.6045532, 261.1123657, -394.9813843, 410.2364197
3: -202.3011475, 285.7830200, -182.2870636, 257.7313843, -460.0325317, 468.0700073
4: -161.6605377, 306.3764954, -145.5399628, 276.2418518, -437.9023132, 451.9163818

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_B1_A2_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -196.9746552, 293.2734375, -177.6839905, 264.9102783, -461.8849182, 470.9574280
1: -154.8932800, 280.6892090, -139.8705139, 253.5063019, -408.3995667, 420.5597229
2: -133.8690796, 289.6318665, -120.8578186, 261.5688171, -395.4378357, 410.4896545
3: -202.3011475, 285.7830200, -182.6897888, 258.0085449, -460.3096313, 468.4727783
4: -161.6605377, 306.3764954, -145.8274994, 276.7703552, -438.4308167, 452.2039795

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

## BFS NS instance: NS_B1_A2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -197.6980591, 294.0541992, -177.1547394, 264.3292236, -462.0272827, 471.2088318
1: -155.3034973, 281.1775818, -139.6272736, 253.1960449, -408.4995422, 420.8048096
2: -134.2444611, 290.2764587, -120.6045532, 261.1123657, -395.3568115, 410.8810120
3: -202.9371185, 286.2823792, -182.2870636, 257.7313843, -460.6685181, 468.5693359
4: -162.0886688, 307.1148987, -145.5399628, 276.2418518, -438.3305054, 452.6547852

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_B1_A2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -197.6980591, 294.0541992, -177.6839905, 264.9102783, -462.6083374, 471.7381592
1: -155.3034973, 281.1775818, -139.8705139, 253.5063019, -408.8098145, 421.0480957
2: -134.2444611, 290.2764587, -120.8578186, 261.5688171, -395.8132629, 411.1342468
3: -202.9371185, 286.2823792, -182.6897888, 258.0085449, -460.9456482, 468.9721069
4: -162.0886688, 307.1148987, -145.8274994, 276.7703552, -438.8590088, 452.9423828

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## BFS NS instance: NS_B2_A1_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -84.8147354, 128.6466370, -98.8425827, 148.5214691, -233.3362122, 227.4892120
1: -67.6151733, 124.5813828, -78.1776886, 143.6714935, -211.2866516, 202.7590637
2: -58.4463959, 128.0579529, -67.6280899, 147.9380035, -206.3843994, 195.6860199
3: -88.3538437, 125.8761673, -102.3993759, 145.3697205, -233.7235413, 228.2755127
4: -70.3067856, 135.6326599, -81.4340668, 156.7362823, -227.0430603, 217.0667114

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_B1_A1_A1_B1_A1

### Relational analysis result of NS_B2_A1_B1_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4094795, upper bound: 339.4079511
time: 1.17 seconds

## Relational analysis of NS_B2_A1_B1_A1_A1_B1_A2

### Relational analysis result of NS_B2_A1_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4215167, upper bound: 339.4132048
time: 0.96 seconds

## BFS NS instance: NS_B2_A1_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -84.8147354, 128.6466370, -184.0684052, 276.5903320, -361.4050598, 312.7150269
1: -67.6151733, 124.5813828, -145.6859741, 267.2598267, -334.8750000, 270.2672729
2: -58.4463959, 128.0579529, -126.0879517, 274.5840149, -333.0303345, 254.1458740
3: -88.3538437, 125.8761673, -190.1156464, 270.6260376, -358.9798279, 315.9917603
4: -70.3067856, 135.6326599, -151.7548981, 290.9830627, -361.2898560, 287.3875732

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 9

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_B1_A1_A1_B2_A1

### Relational analysis result of NS_B2_A1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4229320, upper bound: 339.4105599
time: 1.01 seconds

## Relational analysis of NS_B2_A1_B1_A1_A1_B2_A2

### Relational analysis result of NS_B2_A1_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4238138, upper bound: 339.4174798
time: 0.97 seconds

## BFS NS instance: NS_B2_A1_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -176.4945679, 266.0788269, -98.8425827, 148.5214691, -325.0160217, 364.9213562
1: -140.1070404, 257.6560364, -78.1776886, 143.6714935, -283.7785339, 335.8337402
2: -121.1961670, 264.3083801, -67.6280899, 147.9380035, -269.1341553, 331.9364624
3: -182.6568146, 260.6510925, -102.3993759, 145.3697205, -328.0265503, 363.0504761
4: -145.7884064, 280.0209351, -81.4340668, 156.7362823, -302.5246582, 361.4549866

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_B1_A1_A2_B1_A1

### Relational analysis result of NS_B2_A1_B1_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4103801, upper bound: 339.4136285
time: 0.68 seconds

## Relational analysis of NS_B2_A1_B1_A1_A2_B1_A2

### Relational analysis result of NS_B2_A1_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4164766, upper bound: 339.4153465
time: 0.73 seconds

## BFS NS instance: NS_B2_A1_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -176.4945679, 266.0788269, -184.0684052, 276.5903320, -453.0848999, 450.1472168
1: -140.1070404, 257.6560364, -145.6859741, 267.2598267, -407.3668823, 403.3419800
2: -121.1961670, 264.3083801, -126.0879517, 274.5840149, -395.7801819, 390.3963318
3: -182.6568146, 260.6510925, -190.1156464, 270.6260376, -453.2828369, 450.7666626
4: -145.7884064, 280.0209351, -151.7548981, 290.9830627, -436.7714539, 431.7758179

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B2_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_B1_A1_A2_B2_A1

### Relational analysis result of NS_B2_A1_B1_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4103801, upper bound: 339.4143733
time: 0.67 seconds

## Relational analysis of NS_B2_A1_B1_A1_A2_B2_A2

### Relational analysis result of NS_B2_A1_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4164766, upper bound: 339.4160914
time: 0.96 seconds

## BFS NS instance: NS_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -184.0684052, 276.5903320, -98.8425827, 148.5214691, -332.5898743, 375.4328613
1: -145.6859741, 267.2598267, -78.1776886, 143.6714935, -289.3574219, 345.4375000
2: -126.0879517, 274.5840149, -67.6280899, 147.9380035, -274.0259399, 342.2120667
3: -190.1156464, 270.6260376, -102.3993759, 145.3697205, -335.4853210, 373.0254211
4: -151.7548981, 290.9830627, -81.4340668, 156.7362823, -308.4911804, 372.4171143

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_A1_B1_A2_B1_A2_A1

### Relational analysis result of NS_B2_A1_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4155606, upper bound: 339.4171913
time: 1.01 seconds

## Relational analysis of NS_B2_A1_B1_A2_B1_A2_A2

### Relational analysis result of NS_B2_A1_B1_A2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4095083, upper bound: 339.4153859
time: 0.87 seconds

## BFS NS instance: NS_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -98.8425827, 148.5214691, -184.0684052, 276.5903320, -375.4328613, 332.5898743
1: -78.1776886, 143.6714935, -145.6859741, 267.2598267, -345.4375000, 289.3574219
2: -67.6280899, 147.9380035, -126.0879517, 274.5840149, -342.2120667, 274.0259399
3: -102.3993759, 145.3697205, -190.1156464, 270.6260376, -373.0254211, 335.4853210
4: -81.4340668, 156.7362823, -151.7548981, 290.9830627, -372.4171143, 308.4911804

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_B2_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4086646, upper bound: 339.4156955
time: 1.02 seconds

## Relational analysis of NS_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_B2_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4087748, upper bound: 339.4087355
time: 1.03 seconds

## BFS NS instance: NS_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -184.0684052, 276.5903320, -184.0684052, 276.5903320, -460.6587524, 460.6587524
1: -145.6859741, 267.2598267, -145.6859741, 267.2598267, -412.9457703, 412.9457703
2: -126.0879517, 274.5840149, -126.0879517, 274.5840149, -400.6719666, 400.6719666
3: -190.1156464, 270.6260376, -190.1156464, 270.6260376, -460.7416077, 460.7416077
4: -151.7548981, 290.9830627, -151.7548981, 290.9830627, -442.7379761, 442.7379761

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_A1_B1_A2_B2_A2_A1

### Relational analysis result of NS_B2_A1_B1_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4094809, upper bound: 339.4111009
time: 1.06 seconds

## Relational analysis of NS_B2_A1_B1_A2_B2_A2_A2

### Relational analysis result of NS_B2_A1_B1_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4087748, upper bound: 339.4092282
time: 0.92 seconds

## BFS NS instance: NS_B2_A1_B2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -119.7156830, 180.5114136, -198.2290649, 294.8228760, -414.5385742, 378.7404785
1: -95.3078690, 173.8383484, -155.7336426, 281.9318237, -377.2396851, 329.5719604
2: -82.1589203, 178.6991577, -134.6158752, 291.0564270, -373.2153320, 313.3150330
3: -124.4761047, 176.6610718, -203.4941406, 287.0607605, -411.5368652, 380.1551514
4: -98.9478302, 189.1291656, -162.5386810, 307.9455872, -406.8933716, 351.6677551

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A1_B2_A1_A1_A1_B1

### Relational analysis result of NS_B2_A1_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4269406, upper bound: 339.4129993
time: 1.07 seconds

## Relational analysis of NS_B2_A1_B2_A1_A1_A1_B2

### Relational analysis result of NS_B2_A1_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4269407, upper bound: 339.4188100
time: 0.74 seconds

## BFS NS instance: NS_B2_A1_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -118.9497910, 179.1519012, -198.2290649, 294.8228760, -413.7726440, 377.3809814
1: -94.5011826, 172.4779816, -155.7336426, 281.9318237, -376.4329834, 328.2115479
2: -81.5044937, 177.3490448, -134.6158752, 291.0564270, -372.5608826, 311.9649048
3: -123.5130386, 175.2167664, -203.4941406, 287.0607605, -410.5737915, 378.7108765
4: -98.1219101, 187.7532043, -162.5386810, 307.9455872, -406.0675049, 350.2918396

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A1_B2_A1_A1_A2_B1

### Relational analysis result of NS_B2_A1_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4290487, upper bound: 339.4188715
time: 1.07 seconds

## Relational analysis of NS_B2_A1_B2_A1_A1_A2_B2

### Relational analysis result of NS_B2_A1_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4290488, upper bound: 339.4232782
time: 0.95 seconds

## BFS NS instance: NS_B2_A1_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -135.0331573, 202.9118500, -190.5328827, 283.1501160, -418.1832275, 393.4447021
1: -106.9751587, 194.9790344, -149.6785889, 270.6842041, -377.6593628, 344.6576233
2: -92.2772522, 200.5935364, -129.3718872, 279.5513000, -371.8284607, 329.9653931
3: -139.9173431, 198.1777802, -195.6701355, 275.6775513, -415.5948486, 393.8479004
4: -111.1874847, 212.3526459, -156.2226410, 295.8223877, -407.0098877, 368.5752869

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_A1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A1_B2_A1_A2_B1_A1

### Relational analysis result of NS_B2_A1_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4282827, upper bound: 339.4203682
time: 1.33 seconds

## Relational analysis of NS_B2_A1_B2_A1_A2_B1_A2

### Relational analysis result of NS_B2_A1_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4283145, upper bound: 339.4202746
time: 0.99 seconds

## BFS NS instance: NS_B2_A1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -135.0331573, 202.9118500, -201.8305817, 299.9625854, -434.9957275, 404.7423706
1: -106.9751587, 194.9790344, -158.4784241, 286.6781616, -393.6533203, 353.4574585
2: -92.2772522, 200.5935364, -136.9770660, 296.0209961, -388.2981567, 337.5705566
3: -139.9173431, 198.1777802, -207.1752930, 292.0406494, -431.9579773, 405.3530273
4: -111.1874847, 212.3526459, -165.4182587, 313.2349548, -424.4224243, 377.7709045

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A1_B2_A1_A2_B2_A1

### Relational analysis result of NS_B2_A1_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4282827, upper bound: 339.4248525
time: 1.02 seconds

## Relational analysis of NS_B2_A1_B2_A1_A2_B2_A2

### Relational analysis result of NS_B2_A1_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4283145, upper bound: 339.4245940
time: 1.05 seconds

## BFS NS instance: NS_B2_A1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -131.0926666, 196.4400330, -190.5328827, 283.1501160, -414.2427979, 386.9729004
1: -103.5670242, 189.1084595, -149.6785889, 270.6842041, -374.2512207, 338.7870483
2: -89.4489365, 194.6710510, -129.3718872, 279.5513000, -369.0001526, 324.0429077
3: -135.4604492, 192.0638123, -195.6701355, 275.6775513, -411.1379700, 387.7339478
4: -107.7420044, 206.1067505, -156.2226410, 295.8223877, -403.5643921, 362.3293762

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_B2_A1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -131.0926666, 196.4400330, -201.8305817, 299.9625854, -431.0552368, 398.2705994
1: -103.5670242, 189.1084595, -158.4784241, 286.6781616, -390.2451782, 347.5868835
2: -89.4489365, 194.6710510, -136.9770660, 296.0209961, -385.4698486, 331.6480713
3: -135.4604492, 192.0638123, -207.1752930, 292.0406494, -427.5010986, 399.2391052
4: -107.7420044, 206.1067505, -165.4182587, 313.2349548, -420.9769592, 371.5250244

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_B2_A1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -143.4188995, 214.6302185, -190.5328827, 283.1501160, -426.5690002, 405.1630859
1: -113.1838760, 206.2543030, -149.6785889, 270.6842041, -383.8680725, 355.9328613
2: -97.7532501, 212.4093628, -129.3718872, 279.5513000, -377.3044434, 341.7812500
3: -148.1632385, 209.6267853, -195.6701355, 275.6775513, -423.8407898, 405.2969360
4: -117.8254166, 224.9079895, -156.2226410, 295.8223877, -413.6477966, 381.1306152

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_B2_A1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -143.4188995, 214.6302185, -201.8305817, 299.9625854, -443.3814697, 416.4607849
1: -113.1838760, 206.2543030, -158.4784241, 286.6781616, -399.8620300, 364.7327271
2: -97.7532501, 212.4093628, -136.9770660, 296.0209961, -393.7741394, 349.3864136
3: -148.1632385, 209.6267853, -207.1752930, 292.0406494, -440.2038879, 416.8020630
4: -117.8254166, 224.9079895, -165.4182587, 313.2349548, -431.0603638, 390.3262329

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B2_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -177.1547394, 264.3292236, -196.9746552, 293.2734375, -470.4281006, 461.3038635
1: -139.6272736, 253.1960449, -154.8932800, 280.6892090, -420.3164673, 408.0893250
2: -120.6045532, 261.1123657, -133.8690796, 289.6318665, -410.2364197, 394.9813843
3: -182.2870636, 257.7313843, -202.3011475, 285.7830200, -468.0700073, 460.0325317
4: -145.5399628, 276.2418518, -161.6605377, 306.3764954, -451.9163818, 437.9023132

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

## BFS NS instance: NS_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -177.6839905, 264.9102783, -196.9746552, 293.2734375, -470.9574280, 461.8849182
1: -139.8705139, 253.5063019, -154.8932800, 280.6892090, -420.5597229, 408.3995667
2: -120.8578186, 261.5688171, -133.8690796, 289.6318665, -410.4896545, 395.4378357
3: -182.6897888, 258.0085449, -202.3011475, 285.7830200, -468.4727783, 460.3096619
4: -145.8274994, 276.7703552, -161.6605377, 306.3764954, -452.2039795, 438.4308167

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -177.1547394, 264.3292236, -197.6980591, 294.0541992, -471.2088318, 462.0272827
1: -139.6272736, 253.1960449, -155.3034973, 281.1775818, -420.8048096, 408.4995422
2: -120.6045532, 261.1123657, -134.2444611, 290.2764587, -410.8810120, 395.3568115
3: -182.2870636, 257.7313843, -202.9371185, 286.2823792, -468.5693359, 460.6685181
4: -145.5399628, 276.2418518, -162.0886688, 307.1148987, -452.6547852, 438.3305054

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

## BFS NS instance: NS_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -177.6839905, 264.9102783, -197.6980591, 294.0541992, -471.7381592, 462.6083374
1: -139.8705139, 253.5063019, -155.3034973, 281.1775818, -421.0480957, 408.8098145
2: -120.8578186, 261.5688171, -134.2444611, 290.2764587, -411.1342468, 395.8132629
3: -182.6897888, 258.0085449, -202.9371185, 286.2823792, -468.9721069, 460.9456482
4: -145.8274994, 276.7703552, -162.0886688, 307.1148987, -452.9423828, 438.8590088

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## BFS NS instance: NS_B2_A2_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -196.9746552, 293.2734375, -194.6699677, 289.5197144, -486.4943542, 487.9433594
1: -154.8932800, 280.6892090, -152.9782867, 276.8675842, -431.7608643, 433.6674805
2: -133.8690796, 289.6318665, -132.2251282, 285.8564148, -419.7254333, 421.8569946
3: -202.3011475, 285.7830200, -199.9236298, 281.8858643, -484.1869812, 485.7066650
4: -161.6605377, 306.3764954, -159.6447906, 302.4653320, -464.1257935, 466.0213013

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_A2_B2_A2_A1_B1_A1

### Relational analysis result of NS_B2_A2_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4197965, upper bound: 339.4190144
time: 1.04 seconds

## Relational analysis of NS_B2_A2_B2_A2_A1_B1_A2

### Relational analysis result of NS_B2_A2_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4197965, upper bound: 339.4198564
time: 0.91 seconds

## BFS NS instance: NS_B2_A2_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -195.8173981, 291.5317993, -195.5372009, 291.1664734, -486.9837952, 487.0690002
1: -153.9943695, 279.0253296, -153.8409576, 278.8534851, -432.8478394, 432.8662720
2: -133.0907440, 287.9253235, -132.9911041, 287.7222900, -420.8129883, 420.9163208
3: -201.1362610, 284.0896606, -200.9433899, 283.6850586, -484.8213196, 485.0329895
4: -160.7219238, 304.5780029, -160.5455933, 304.3817749, -465.1036987, 465.1235352

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_A2_B2_A2_A1_B2_A1

### Relational analysis result of NS_B2_A2_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4200332, upper bound: 339.4196100
time: 1.00 seconds

## Relational analysis of NS_B2_A2_B2_A2_A1_B2_A2

### Relational analysis result of NS_B2_A2_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4200332, upper bound: 339.4196100
time: 0.92 seconds

## BFS NS instance: NS_B2_A2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -197.6980591, 294.0541992, -194.6699677, 289.5197144, -487.2177734, 488.7240906
1: -155.3034973, 281.1775818, -152.9782867, 276.8675842, -432.1710815, 434.1558838
2: -134.2444611, 290.2764587, -132.2251282, 285.8564148, -420.1008606, 422.5015869
3: -202.9371185, 286.2823792, -199.9236298, 281.8858643, -484.8229980, 486.2059937
4: -162.0886688, 307.1148987, -159.6447906, 302.4653320, -464.5539856, 466.7597046

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_A2_B2_A2_A2_B1_A1

### Relational analysis result of NS_B2_A2_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4198132, upper bound: 339.4196813
time: 1.03 seconds

## Relational analysis of NS_B2_A2_B2_A2_A2_B1_A2

### Relational analysis result of NS_B2_A2_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4198132, upper bound: 339.4197757
time: 0.76 seconds

## BFS NS instance: NS_B2_A2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -196.5220642, 292.2849731, -195.5372009, 291.1664734, -487.6885376, 487.8221741
1: -154.3892517, 279.4899292, -153.8409576, 278.8534851, -433.2426758, 433.3308716
2: -133.4534912, 288.5449524, -132.9911041, 287.7222900, -421.1757812, 421.5359497
3: -201.7515869, 284.5650940, -200.9433899, 283.6850586, -485.4366455, 485.5084534
4: -161.1356964, 305.2891846, -160.5455933, 304.3817749, -465.5174255, 465.8347473

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_A2_B2_A2_A2_B2_A1

### Relational analysis result of NS_B2_A2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4200499, upper bound: 339.4203363
time: 1.01 seconds

## Relational analysis of NS_B2_A2_B2_A2_A2_B2_A2

### Relational analysis result of NS_B2_A2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4200499, upper bound: 339.4205466
time: 0.83 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.54 seconds
NS_B1_A1_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4199883, upper bound: 339.4138910
NS_B1_A1_B1_A1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4156850, upper bound: 339.4133167
NS_B1_A1_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4229320, upper bound: 339.4116518
NS_B1_A1_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4231019, upper bound: 339.4178401
NS_B1_A1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4116518, upper bound: 339.4181886
NS_B1_A1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4164740, upper bound: 339.4180384
NS_B1_A1_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4162512, upper bound: 339.4171973
NS_B1_A1_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4156101, upper bound: 339.4171973
NS_B1_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4079511, upper bound: 339.4094795
NS_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4132048, upper bound: 339.4215167
NS_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4099035, upper bound: 339.4242462
NS_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4168246, upper bound: 339.4251699
NS_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4133689, upper bound: 339.4171054
NS_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4133689, upper bound: 339.4170480
NS_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4133689, upper bound: 339.4204931
NS_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4133689, upper bound: 339.4200797
NS_B1_A1_B2_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4213997, upper bound: 339.4191128
NS_B1_A1_B2_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4211617, upper bound: 339.4205840
NS_B1_A1_B2_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4214701, upper bound: 339.4203280
NS_B1_A1_B2_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4211303, upper bound: 339.4200758
NS_B1_A2_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4242693, upper bound: 339.4300457
NS_B1_A2_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4242693, upper bound: 339.4300457
NS_B1_A2_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4228289, upper bound: 339.4300457
NS_B1_A2_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4228289, upper bound: 339.4300457
NS_B1_A2_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4129993, upper bound: 339.4269406
NS_B1_A2_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4129993, upper bound: 339.4313783
NS_B1_A2_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4188715, upper bound: 339.4290488
NS_B1_A2_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4188715, upper bound: 339.4313220
NS_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4129993, upper bound: 339.4282826
NS_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4188715, upper bound: 339.4283144
NS_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4129993, upper bound: 339.4269406
NS_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4188715, upper bound: 339.4323279
NS_B2_A1_B1_A1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4094795, upper bound: 339.4079511
NS_B2_A1_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4215167, upper bound: 339.4132048
NS_B2_A1_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4229320, upper bound: 339.4105599
NS_B2_A1_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4238138, upper bound: 339.4174798
NS_B2_A1_B1_A1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4103801, upper bound: 339.4136285
NS_B2_A1_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4164766, upper bound: 339.4153465
NS_B2_A1_B1_A1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4103801, upper bound: 339.4143733
NS_B2_A1_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4164766, upper bound: 339.4160914
NS_B2_A1_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4155606, upper bound: 339.4171913
NS_B2_A1_B1_A2_B1_A2_A2, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4095083, upper bound: 339.4153859
NS_B2_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4086646, upper bound: 339.4156955
NS_B2_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4087748, upper bound: 339.4087355
NS_B2_A1_B1_A2_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4094809, upper bound: 339.4111009
NS_B2_A1_B1_A2_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4087748, upper bound: 339.4092282
NS_B2_A1_B2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4269406, upper bound: 339.4129993
NS_B2_A1_B2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4269407, upper bound: 339.4188100
NS_B2_A1_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4290487, upper bound: 339.4188715
NS_B2_A1_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4290488, upper bound: 339.4232782
NS_B2_A1_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4282827, upper bound: 339.4203682
NS_B2_A1_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4283145, upper bound: 339.4202746
NS_B2_A1_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4282827, upper bound: 339.4248525
NS_B2_A1_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4283145, upper bound: 339.4245940
NS_B2_A2_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4197965, upper bound: 339.4190144
NS_B2_A2_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4197965, upper bound: 339.4198564
NS_B2_A2_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4200332, upper bound: 339.4196100
NS_B2_A2_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4200332, upper bound: 339.4196100
NS_B2_A2_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4198132, upper bound: 339.4196813
NS_B2_A2_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4198132, upper bound: 339.4197757
NS_B2_A2_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4200499, upper bound: 339.4203363
NS_B2_A2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 4, lower bound: -339.4200499, upper bound: 339.4205466

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.51 + 417.50 = 421.01 seconds
