## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 1)
Time budget: 420 seconds
Split limit: 100
Threshold: 380.961918313704


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-120.7598648, 310.2168884, -120.7598648, 310.2168884, -430.9767456, 430.9767456)
1: (-306.0167542, 469.4156189, -306.0167542, 469.4156189, -775.4323730, 775.4323730)
2: (-197.4686890, 458.7374573, -197.4686890, 458.7374573, -656.2061768, 656.2061768)
3: (-330.1389465, 542.4745483, -330.1389465, 542.4745483, -872.6135254, 872.6134644)
4: (-287.0395203, 524.3436890, -287.0395203, 524.3436890, -811.3831177, 811.3831177)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.74 + 1.96 = 2.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -380.9771574, upper bound: 380.9771574

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9766519, upper bound: 380.9770014
time: 0.66 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9765270, upper bound: 380.9765270
time: 1.02 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.76 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.76
Output dim: 0, lower bound: -380.9766519, upper bound: 380.9770014
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.76
Output dim: 0, lower bound: -380.9765270, upper bound: 380.9765270

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -115.6930923, 296.1134949, -117.4032059, 300.8791199, -416.5722046, 413.5166626
1: -293.0917053, 447.5654907, -297.4370728, 454.9557800, -748.0474854, 745.0025635
2: -189.2838745, 437.6819153, -192.0356445, 444.7875671, -634.0714111, 629.7175293
3: -316.2052917, 517.3720093, -320.9125671, 525.8545532, -842.0597534, 838.2845459
4: -275.2095947, 499.8500061, -279.2066345, 508.1421509, -783.3517456, 779.0565186

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9763536, upper bound: 380.9769175
time: 0.79 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9577696, upper bound: 380.9615480
time: 0.69 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -125.1845093, 320.7346802, -118.0939941, 303.5114136, -428.6958618, 438.8286438
1: -316.3167725, 484.9741821, -299.1312866, 459.4036865, -775.7204590, 784.1054077
2: -205.2873840, 474.5839844, -193.0708160, 448.6973572, -653.9846191, 667.6547852
3: -341.5942383, 560.5359497, -322.8647766, 530.8626099, -872.4568481, 883.4006348
4: -298.0682678, 541.2152710, -280.7765198, 512.9891357, -811.0573730, 821.9918213

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9709730, upper bound: 380.9733305
time: 0.66 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9708210, upper bound: 380.9708210
time: 0.73 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.12 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 0, lower bound: -380.9763536, upper bound: 380.9769175
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.12
Output dim: 0, lower bound: -380.9577696, upper bound: 380.9615480
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 0, lower bound: -380.9709730, upper bound: 380.9733305
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 0, lower bound: -380.9708210, upper bound: 380.9708210

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -111.0715790, 284.4209290, -108.2714462, 277.7646790, -388.8362427, 392.6923828
1: -281.1555481, 430.0823669, -274.4936523, 420.0239258, -701.1794434, 704.5760498
2: -181.6007996, 419.9787598, -177.3282623, 410.3757935, -591.9765015, 597.3070068
3: -303.6622314, 497.1092224, -296.1408081, 485.5923462, -789.2545776, 793.2500000
4: -264.4497375, 480.0018005, -257.7760620, 469.0232849, -733.4730225, 737.7775879

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9763274, upper bound: 380.9769175
time: 0.95 seconds

## Relational analysis of NS_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9763274, upper bound: 380.9769175
time: 0.68 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -122.5452347, 313.2480774, -112.9574814, 289.4706116, -412.0158386, 426.2055664
1: -309.4214783, 473.4932861, -285.8626709, 437.8786316, -747.3001099, 759.3558960
2: -201.0377808, 463.5462341, -184.7526245, 427.9145813, -628.9523926, 648.2988281
3: -334.1148987, 547.5115967, -308.5581055, 506.3545837, -840.4694824, 856.0697021
4: -291.8476257, 528.3049927, -268.7242126, 488.8791504, -780.7266846, 797.0291748

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9709730, upper bound: 380.9715398
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9709730, upper bound: 380.9733286
time: 0.60 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -122.3089600, 313.5388184, -135.3437653, 346.4349365, -468.7438965, 448.8825684
1: -308.8479614, 474.3057556, -341.8498535, 522.4492798, -831.2971802, 816.1555786
2: -200.5241241, 463.8295593, -221.7123871, 512.8471069, -713.3712158, 685.5418701
3: -333.7530212, 548.0718384, -368.9022217, 604.7451172, -938.4981079, 916.9740601
4: -291.4149475, 529.0875244, -321.2390747, 584.3275146, -875.7424316, 850.3264160

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9700780, upper bound: 380.9708210
time: 0.64 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9707874, upper bound: 380.9707874
time: 0.99 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.14 seconds
NS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 0, lower bound: -380.9763274, upper bound: 380.9769175
NS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 0, lower bound: -380.9763274, upper bound: 380.9769175
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 0, lower bound: -380.9709730, upper bound: 380.9715398
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 0, lower bound: -380.9709730, upper bound: 380.9733286
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 0, lower bound: -380.9700780, upper bound: 380.9708210
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 0, lower bound: -380.9707874, upper bound: 380.9707874

## BFS NS instance: NS_A1_B1_B1

### Backsubstitution after applying NS history:
0: -111.0715790, 284.4209290, -106.3006897, 272.3011475, -383.3727417, 390.7215576
1: -281.1555481, 430.0823669, -269.3795471, 411.5380554, -692.6935425, 699.4619141
2: -181.6007996, 419.9787598, -174.1498566, 402.2122192, -583.8129883, 594.1286011
3: -303.6622314, 497.1092224, -290.6614075, 475.8457642, -779.5079956, 787.7706299
4: -264.4497375, 480.0018005, -253.2281799, 459.4552917, -723.9050293, 733.2299194

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9763200, upper bound: 380.9763548
time: 0.64 seconds

## Relational analysis of NS_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9763200, upper bound: 380.9769175
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_B2

### Backsubstitution after applying NS history:
0: -111.0715790, 284.4209290, -129.2013397, 328.0920105, -439.1635742, 413.6222534
1: -281.1555481, 430.0823669, -324.7097168, 495.0908203, -776.2463379, 754.7919922
2: -181.6007996, 419.9787598, -212.4113464, 484.0606689, -665.6614990, 632.3901367
3: -303.6622314, 497.1092224, -350.6284180, 572.7595825, -876.4218140, 847.7376709
4: -264.4497375, 480.0018005, -308.4446106, 552.3624878, -816.8122559, 788.4462891

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9763200, upper bound: 380.9763548
time: 0.88 seconds

## Relational analysis of NS_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9763200, upper bound: 380.9769175
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -117.4251404, 300.8029785, -110.4106064, 282.9319153, -400.3570251, 411.2135925
1: -297.3891602, 454.8566589, -279.4989014, 428.0126343, -725.4017944, 734.3554688
2: -192.9058685, 445.8417664, -180.6282806, 418.4252625, -611.3311157, 626.4700317
3: -319.9935608, 526.4856567, -301.5789490, 494.9932556, -814.9868164, 828.0645752
4: -279.5570374, 508.6053162, -262.6237793, 477.9887390, -757.5457764, 771.2290649

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9709513, upper bound: 380.9715398
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9709513, upper bound: 380.9715398
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -121.1613464, 309.6445618, -112.1142654, 287.2758484, -408.4371948, 421.7587891
1: -305.8045044, 468.1299438, -283.6395264, 434.6206055, -740.4249878, 751.7694702
2: -198.6930847, 458.2022705, -183.3278961, 424.6378784, -623.3309326, 641.5300293
3: -330.3354797, 541.1992798, -306.2512207, 502.5166626, -832.8521729, 847.4503784
4: -288.5749207, 522.2458496, -266.7615051, 485.1656494, -773.7406006, 789.0073242

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9676992, upper bound: 380.9701501
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9709730, upper bound: 380.9733286
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: -120.1552124, 307.8955078, -130.1763153, 333.6618652, -453.8170776, 438.0718079
1: -303.3125000, 465.8132019, -329.9284668, 503.3043518, -806.6168213, 795.7416992
2: -197.0559845, 455.6010437, -213.6962128, 494.4724426, -691.5284424, 669.2972412
3: -327.6557617, 538.2821045, -355.0840149, 583.1473389, -910.8031006, 893.3660889
4: -286.3276672, 519.5578003, -308.9260864, 564.0195312, -850.3471069, 828.4838867

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9700780, upper bound: 380.9708210
time: 0.80 seconds

## Relational analysis of NS_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9700780, upper bound: 380.9708210
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: -121.5113754, 311.4691162, -134.1880035, 343.4147034, -464.9260864, 445.6570740
1: -306.7639771, 471.2208862, -338.8255920, 517.9617310, -824.7256470, 810.0465088
2: -199.1764679, 460.7610474, -219.7531738, 508.3778687, -707.5543213, 680.5142212
3: -331.5754395, 544.4447632, -365.7257385, 599.4548950, -931.0303345, 910.1705322
4: -289.5316162, 525.5970459, -318.5069580, 579.2708130, -868.8024292, 844.1040039

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9707874, upper bound: 380.9700780
time: 0.63 seconds

## Relational analysis of NS_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9707874, upper bound: 380.9707874
time: 0.73 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.31 seconds
NS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.31
Output dim: 0, lower bound: -380.9763200, upper bound: 380.9763548
NS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.31
Output dim: 0, lower bound: -380.9763200, upper bound: 380.9769175
NS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.31
Output dim: 0, lower bound: -380.9763200, upper bound: 380.9763548
NS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.31
Output dim: 0, lower bound: -380.9763200, upper bound: 380.9769175
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.31
Output dim: 0, lower bound: -380.9709513, upper bound: 380.9715398
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.31
Output dim: 0, lower bound: -380.9709513, upper bound: 380.9715398
NS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.31
Output dim: 0, lower bound: -380.9676992, upper bound: 380.9701501
NS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.31
Output dim: 0, lower bound: -380.9709730, upper bound: 380.9733286
NS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.31
Output dim: 0, lower bound: -380.9700780, upper bound: 380.9708210
NS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.31
Output dim: 0, lower bound: -380.9700780, upper bound: 380.9708210
NS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.31
Output dim: 0, lower bound: -380.9707874, upper bound: 380.9700780
NS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.31
Output dim: 0, lower bound: -380.9707874, upper bound: 380.9707874

## BFS NS instance: NS_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -106.3006897, 272.3011475, -106.3006897, 272.3011475, -378.6018066, 378.6018066
1: -269.3795471, 411.5380554, -269.3795471, 411.5380554, -680.9176025, 680.9176025
2: -174.1498566, 402.2122192, -174.1498566, 402.2122192, -576.3620605, 576.3620605
3: -290.6614075, 475.8457642, -290.6614075, 475.8457642, -766.5072021, 766.5072021
4: -253.2281799, 459.4552917, -253.2281799, 459.4552917, -712.6834717, 712.6834717

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_B1_A1_B1

### Relational analysis result of NS_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9763926, upper bound: 380.9764540
time: 0.64 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2

### Relational analysis result of NS_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9764540, upper bound: 380.9764540
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -113.4501877, 290.3067627, -106.3006897, 272.3011475, -385.7513428, 396.6073303
1: -287.3683777, 439.0522766, -269.3795471, 411.5380554, -698.9064331, 708.4318237
2: -185.6127930, 429.0042114, -174.1498566, 402.2122192, -587.8250122, 603.1539917
3: -310.2089233, 507.4839478, -290.6614075, 475.8457642, -786.0546265, 798.1453247
4: -269.9965210, 489.9992981, -253.2281799, 459.4552917, -729.4517822, 743.2274780

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_B1_A2_A1

### Relational analysis result of NS_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9764540, upper bound: 380.9771490
time: 0.77 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2

### Relational analysis result of NS_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9764540, upper bound: 380.9771490
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -106.3006897, 272.3011475, -129.2013397, 328.0920105, -434.3926086, 401.5025024
1: -269.3795471, 411.5380554, -324.7097168, 495.0908203, -764.4703369, 736.2476196
2: -174.1498566, 402.2122192, -212.4113464, 484.0606689, -658.2105103, 614.6235352
3: -290.6614075, 475.8457642, -350.6284180, 572.7595825, -863.4210205, 826.4741821
4: -253.2281799, 459.4552917, -308.4446106, 552.3624878, -805.5906982, 767.8999023

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_B2_A1_B1

### Relational analysis result of NS_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9762363, upper bound: 380.9763545
time: 0.83 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2

### Relational analysis result of NS_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9763482, upper bound: 380.9763548
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -113.4501877, 290.3067627, -129.2013397, 328.0920105, -441.5422058, 419.5080872
1: -287.3683777, 439.0522766, -324.7097168, 495.0908203, -782.4592285, 763.7619019
2: -185.6127930, 429.0042114, -212.4113464, 484.0606689, -669.6734619, 641.4155273
3: -310.2089233, 507.4839478, -350.6284180, 572.7595825, -882.9684448, 858.1123047
4: -269.9965210, 489.9992981, -308.4446106, 552.3624878, -822.3590088, 798.4439087

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_B2_A2_B1

### Relational analysis result of NS_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9762363, upper bound: 380.9769157
time: 0.63 seconds

## Relational analysis of NS_A1_B1_B2_A2_B2

### Relational analysis result of NS_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9763482, upper bound: 380.9769159
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -117.4251404, 300.8029785, -109.5497055, 280.4503174, -397.8754272, 410.3526917
1: -297.3891602, 454.8566589, -278.9008179, 424.2936707, -721.6828613, 733.7573853
2: -192.9058685, 445.8417664, -180.1126099, 414.7514038, -607.6572876, 625.9543457
3: -319.9935608, 526.4856567, -299.5872192, 491.2593384, -811.2529297, 826.0728760
4: -279.5570374, 508.6053162, -260.3315735, 474.6211243, -754.1781616, 768.9368896

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B1_B1

### Relational analysis result of NS_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9707196, upper bound: 380.9712627
time: 0.99 seconds

## Relational analysis of NS_A2_B1_A1_B1_B2

### Relational analysis result of NS_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9705854, upper bound: 380.9709629
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -117.4251404, 300.8029785, -111.5340500, 285.7691040, -403.1942444, 412.3370056
1: -297.3891602, 454.8566589, -282.1060181, 432.3901978, -729.7792969, 736.9625854
2: -192.9058685, 445.8417664, -182.3500214, 422.3811646, -615.2870483, 628.1917725
3: -319.9935608, 526.4856567, -304.6668091, 499.8968506, -819.8903809, 831.1524658
4: -279.5570374, 508.6053162, -265.4165955, 482.6192017, -762.1762695, 774.0217896

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B2_B1

### Relational analysis result of NS_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9707196, upper bound: 380.9712627
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2

### Relational analysis result of NS_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9705854, upper bound: 380.9709629
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -116.6081848, 297.3800354, -109.2321243, 280.0349121, -396.6430969, 406.6121521
1: -294.0276184, 449.4705505, -276.2138367, 423.8153381, -717.8429565, 725.6843872
2: -191.4128113, 440.3003235, -178.5350952, 413.8956299, -605.3082886, 618.8353271
3: -317.4993591, 519.6891479, -298.3203430, 490.0159912, -807.5153198, 818.0094604
4: -277.7133789, 501.5002136, -259.9468689, 473.0108948, -750.7242432, 761.4470215

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_A1_A1

### Relational analysis result of NS_A2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9629113, upper bound: 380.9631134
time: 0.92 seconds

## Relational analysis of NS_A2_B1_A2_A1_A2

### Relational analysis result of NS_A2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9664377, upper bound: 380.9680602
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -120.2597122, 307.3631287, -111.4418564, 285.5944824, -405.8541870, 418.8049622
1: -303.5092163, 464.6740723, -281.9390564, 432.0983582, -735.6075439, 746.6131592
2: -197.2104340, 454.8011780, -182.2091675, 422.1674500, -619.3778076, 637.0103149
3: -327.8020630, 537.2023926, -304.3850403, 499.6018372, -827.4038696, 841.5872803
4: -286.4281006, 518.3792725, -265.1588745, 482.3433838, -768.7714233, 783.5380859

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_A2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9709513, upper bound: 380.9733286
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9709513, upper bound: 380.9733286
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -118.6108322, 302.5816345, -130.1763153, 333.6618652, -452.2726746, 432.7579346
1: -299.2205505, 457.2488403, -329.9284668, 503.3043518, -802.5249023, 787.1773071
2: -194.7111816, 447.9691162, -213.6962128, 494.4724426, -689.1835938, 661.6653442
3: -322.9769287, 528.9791870, -355.0840149, 583.1473389, -906.1242676, 884.0631104
4: -282.5789795, 510.1644897, -308.9260864, 564.0195312, -846.5984497, 819.0905762

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_B1_A1_A1

### Relational analysis result of NS_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9700780, upper bound: 380.9701711
time: 0.70 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2

### Relational analysis result of NS_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9700780, upper bound: 380.9708210
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -150.7656860, 381.8880005, -130.1763153, 333.6618652, -484.4275513, 512.0643311
1: -378.1133118, 574.6561279, -329.9284668, 503.3043518, -881.4176636, 904.5845947
2: -248.1350555, 564.5180054, -213.6962128, 494.4724426, -742.6074829, 778.2142334
3: -407.8330078, 665.9803467, -355.0840149, 583.1473389, -990.9802856, 1021.0642700
4: -358.9593506, 642.6400757, -308.9260864, 564.0195312, -922.9787598, 951.5661621

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_B1_A2_A1

### Relational analysis result of NS_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9700780, upper bound: 380.9701711
time: 0.69 seconds

## Relational analysis of NS_A2_B2_B1_A2_A2

### Relational analysis result of NS_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9700780, upper bound: 380.9708210
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -117.4088974, 301.5146790, -134.1880035, 343.4147034, -460.8236084, 435.7026367
1: -297.4158936, 456.2221680, -338.8255920, 517.9617310, -815.3776245, 795.0477295
2: -192.7341003, 446.7398376, -219.7531738, 508.3778687, -701.1119385, 666.4930420
3: -320.2634583, 527.7058105, -365.7257385, 599.4548950, -919.7182617, 893.4315186
4: -279.5198975, 510.1401367, -318.5069580, 579.2708130, -858.7907104, 828.6470947

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_B2_A1_A1

### Relational analysis result of NS_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9700780, upper bound: 380.9700780
time: 0.67 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2

### Relational analysis result of NS_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9700780, upper bound: 380.9700780
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -121.0105820, 310.1716614, -134.1880035, 343.4147034, -464.4252930, 444.3596191
1: -305.4521484, 469.2937012, -338.8255920, 517.9617310, -823.4138794, 808.1192627
2: -198.3283997, 458.8338318, -219.7531738, 508.3778687, -706.7062988, 678.5870361
3: -330.2074280, 542.1704102, -365.7257385, 599.4548950, -929.6622314, 907.8961182
4: -288.3508301, 523.4086914, -318.5069580, 579.2708130, -867.6215210, 841.9155884

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_B2_A2_A1

### Relational analysis result of NS_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9700780, upper bound: 380.9707874
time: 0.68 seconds

## Relational analysis of NS_A2_B2_B2_A2_A2

### Relational analysis result of NS_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9700780, upper bound: 380.9707874
time: 0.66 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.78 seconds
NS_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -380.9763926, upper bound: 380.9764540
NS_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -380.9764540, upper bound: 380.9764540
NS_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -380.9764540, upper bound: 380.9771490
NS_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -380.9764540, upper bound: 380.9771490
NS_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -380.9762363, upper bound: 380.9763545
NS_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -380.9763482, upper bound: 380.9763548
NS_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -380.9762363, upper bound: 380.9769157
NS_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -380.9763482, upper bound: 380.9769159
NS_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -380.9707196, upper bound: 380.9712627
NS_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -380.9705854, upper bound: 380.9709629
NS_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -380.9707196, upper bound: 380.9712627
NS_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -380.9705854, upper bound: 380.9709629
NS_A2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -380.9629113, upper bound: 380.9631134
NS_A2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -380.9664377, upper bound: 380.9680602
NS_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -380.9709513, upper bound: 380.9733286
NS_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -380.9709513, upper bound: 380.9733286
NS_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -380.9700780, upper bound: 380.9701711
NS_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -380.9700780, upper bound: 380.9708210
NS_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -380.9700780, upper bound: 380.9701711
NS_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -380.9700780, upper bound: 380.9708210
NS_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -380.9700780, upper bound: 380.9700780
NS_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -380.9700780, upper bound: 380.9700780
NS_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -380.9700780, upper bound: 380.9707874
NS_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -380.9700780, upper bound: 380.9707874

## BFS NS instance: NS_A1_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -104.1511765, 266.6576233, -100.9832230, 259.1163940, -363.2675781, 367.6407776
1: -263.8249207, 402.9391479, -257.3857727, 391.7742920, -655.5992432, 660.3249512
2: -170.6576996, 393.9704895, -166.2153931, 383.3316650, -553.9893799, 560.1857910
3: -284.5997314, 465.9592896, -276.1808472, 453.5200500, -738.1196289, 742.1400146
4: -248.1468964, 449.8768311, -240.2730255, 438.6277466, -686.7746582, 690.1496582

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9763926, upper bound: 380.9763926
time: 0.71 seconds

## Relational analysis of NS_A1_B1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9763926, upper bound: 380.9764540
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -105.5417099, 270.3027954, -105.0967255, 269.1310425, -374.6727600, 375.3995361
1: -267.3861389, 408.5657349, -266.2148132, 406.8220520, -674.2081909, 674.7805176
2: -172.8592072, 399.2352905, -172.1022644, 397.4872437, -570.3464355, 571.3374634
3: -288.5862122, 472.3426819, -287.3706360, 470.2888489, -758.8749390, 759.7133179
4: -251.4241333, 456.0984497, -250.3684387, 454.1274109, -705.5515137, 706.4667969

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9764540, upper bound: 380.9763926
time: 0.66 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9764540, upper bound: 380.9764540
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -109.7967453, 280.7104187, -104.1511765, 266.6576233, -376.4542542, 384.8616028
1: -279.6531677, 424.5692444, -263.8249207, 402.9391479, -682.5922852, 688.3941650
2: -180.5979156, 415.0063171, -170.6576996, 393.9704895, -574.5682983, 585.6640015
3: -300.4217224, 491.3782959, -284.5997314, 465.9592896, -766.3809814, 775.9780273
4: -261.0100403, 474.8264465, -248.1468964, 449.8768311, -710.8868408, 722.9733276

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B1_A2_A1_A1

### Relational analysis result of NS_A1_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9735816, upper bound: 380.9726927
time: 0.68 seconds

## Relational analysis of NS_A1_B1_B1_A2_A1_A2

### Relational analysis result of NS_A1_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9722911, upper bound: 380.9720856
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -111.9040985, 286.3507996, -105.5417099, 270.3027954, -382.2068787, 391.8925171
1: -283.2727661, 433.1998596, -267.3861389, 408.5657349, -691.8383789, 700.5859985
2: -182.9956360, 423.0840759, -172.8592072, 399.2352905, -582.2308350, 595.9432373
3: -305.9629822, 500.5824890, -288.5862122, 472.3426819, -778.3056030, 789.1685181
4: -266.4358215, 483.3019714, -251.4241333, 456.0984497, -722.5343018, 734.7260742

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9763964, upper bound: 380.9771490
time: 0.73 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9763964, upper bound: 380.9771490
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -104.1511765, 266.6576233, -121.8744736, 310.4247131, -414.5758972, 388.5319519
1: -263.8249207, 402.9391479, -306.7044678, 468.8989563, -732.7238770, 709.6436157
2: -170.6576996, 393.9704895, -200.2115784, 458.5315552, -629.1892700, 594.1820679
3: -284.5997314, 465.9592896, -330.4346008, 542.6766357, -827.2763062, 796.3939209
4: -248.1468964, 449.8768311, -291.0246887, 523.5110474, -771.6579590, 740.9014893

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B2_A1_B1_B1

### Relational analysis result of NS_A1_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9722074, upper bound: 380.9734466
time: 0.74 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9762363, upper bound: 380.9763048
time: 0.68 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9762363, upper bound: 380.9763545
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -105.5417099, 270.3027954, -128.0279083, 324.9985352, -430.5402527, 398.3306580
1: -267.3861389, 408.5657349, -321.6183167, 490.5220337, -757.9082031, 730.1838379
2: -172.8592072, 399.2352905, -210.4292297, 479.4671936, -652.3264160, 609.6644287
3: -288.5862122, 472.3426819, -347.3976135, 567.3754883, -855.9616089, 819.7402954
4: -251.4241333, 456.0984497, -305.6956177, 547.1682129, -798.5922852, 761.7940674

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_B2_A1_B2_B1

### Relational analysis result of NS_A1_B1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9743737, upper bound: 380.9728527
time: 0.66 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_B2

### Relational analysis result of NS_A1_B1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9763482, upper bound: 380.9763548
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -111.0546417, 284.1024780, -121.8744736, 310.4247131, -421.4793396, 405.9767761
1: -281.4180603, 429.6471252, -306.7044678, 468.8989563, -750.3169556, 736.3514404
2: -181.7767334, 420.0047302, -200.2115784, 458.5315552, -640.3082886, 620.2163086
3: -303.6212769, 496.6630249, -330.4346008, 542.6766357, -846.2978516, 827.0975952
4: -264.2311401, 479.6477051, -291.0246887, 523.5110474, -787.7421875, 770.6723633

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9762453, upper bound: 380.9769157
time: 0.88 seconds

## Relational analysis of NS_A1_B1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9762453, upper bound: 380.9769157
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -112.4264603, 287.6943359, -128.0279083, 324.9985352, -437.4249878, 415.7221680
1: -284.6508789, 435.1862183, -321.6183167, 490.5220337, -775.1729126, 756.8045654
2: -183.8758545, 425.0906982, -210.4292297, 479.4671936, -663.3430176, 635.5198364
3: -307.3930664, 502.9256592, -347.3976135, 567.3754883, -874.7685547, 850.3231812
4: -267.6438904, 485.5726929, -305.6956177, 547.1682129, -814.8121338, 791.2681885

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_B2_A2_B2_B1

### Relational analysis result of NS_A1_B1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9679076, upper bound: 380.9692741
time: 0.67 seconds

## Relational analysis of NS_A1_B1_B2_A2_B2_B2

### Relational analysis result of NS_A1_B1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9762967, upper bound: 380.9768545
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -117.4251404, 300.8029785, -109.4662857, 280.2384338, -397.6635437, 410.2692566
1: -297.3891602, 454.8566589, -278.6857300, 423.9806824, -721.3698730, 733.5422974
2: -192.9058685, 445.8417664, -179.9723053, 414.4336548, -607.3395386, 625.8140869
3: -319.9935608, 526.4856567, -299.3610840, 490.8939514, -810.8875122, 825.8467407
4: -279.5570374, 508.6053162, -260.1336975, 474.2632751, -753.8203125, 768.7389526

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9707196, upper bound: 380.9724326
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A1_B1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9707196, upper bound: 380.9724326
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -117.4251404, 300.8029785, -109.7886734, 281.0440674, -398.4691772, 410.5916443
1: -297.3891602, 454.8566589, -279.5932922, 425.1916199, -722.5808105, 734.4499512
2: -192.9058685, 445.8417664, -180.5375671, 415.6315613, -608.5374146, 626.3793335
3: -319.9935608, 526.4856567, -300.2929993, 492.3227844, -812.3163452, 826.7786865
4: -279.5570374, 508.6053162, -260.8638306, 475.6730042, -755.2300415, 769.4691162

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9705854, upper bound: 380.9721847
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A1_B1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9705854, upper bound: 380.9721847
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -117.4251404, 300.8029785, -111.4585800, 285.5787048, -403.0038147, 412.2615662
1: -297.3891602, 454.8566589, -281.9114075, 432.1082764, -729.4974365, 736.7680054
2: -192.9058685, 445.8417664, -182.2229309, 422.0971375, -615.0029907, 628.0646973
3: -319.9935608, 526.4856567, -304.4627991, 499.5685120, -819.5620728, 830.9484863
4: -279.5570374, 508.6053162, -265.2369995, 482.2981567, -761.8552246, 773.8422852

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9707392, upper bound: 380.9712627
time: 0.92 seconds

## Relational analysis of NS_A2_B1_A1_B2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9707392, upper bound: 380.9712627
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -117.4251404, 300.8029785, -111.7214584, 286.2406921, -403.6658020, 412.5244446
1: -297.3891602, 454.8566589, -282.6607971, 433.1039124, -730.4930420, 737.5174561
2: -192.9058685, 445.8417664, -182.6893921, 423.0865479, -615.9924316, 628.5311279
3: -319.9935608, 526.4856567, -305.2257690, 500.7460022, -820.7395630, 831.7113647
4: -279.5570374, 508.6053162, -265.8354797, 483.4584351, -763.0155029, 774.4407349

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9705951, upper bound: 380.9709629
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9705951, upper bound: 380.9709629
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A1

### Backsubstitution after applying NS history:
0: -112.2395554, 286.7243958, -105.9616699, 271.9833679, -384.2229004, 392.6860657
1: -283.6421204, 432.9821777, -267.9221191, 411.7729797, -695.4149780, 700.9042969
2: -184.5522308, 424.4610901, -173.1523743, 401.7651367, -586.3173828, 597.6134644
3: -306.0482178, 500.7993774, -289.4580383, 476.0325012, -782.0806274, 790.2574463
4: -266.9277649, 483.7980957, -252.1842346, 459.3689270, -726.2966919, 735.9822998

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_A1_A1_A1

### Relational analysis result of NS_A2_B1_A2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9563485, upper bound: 380.9524280
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A2_A1_A1_A2

### Relational analysis result of NS_A2_B1_A2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9597056, upper bound: 380.9604470
time: 0.62 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -115.6801910, 295.0654907, -108.6212006, 278.5201721, -394.2003479, 403.6867065
1: -291.6250916, 446.0173340, -274.6693726, 421.5314026, -713.1564941, 720.6866455
2: -189.8429871, 436.8557739, -177.5199280, 411.6577148, -601.5007324, 614.3755493
3: -314.9836426, 515.6694946, -296.6787109, 487.3547974, -802.3383789, 812.3482056
4: -275.5262756, 497.5929871, -258.4672546, 470.4599915, -745.9862671, 756.0601807

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_A1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9630773, upper bound: 380.9654295
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_A1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9629280, upper bound: 380.9650371
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -120.2597122, 307.3631287, -108.8274078, 278.6454773, -398.9051819, 416.1905212
1: -303.5092163, 464.6740723, -277.0440063, 421.5912781, -725.1004639, 741.7180786
2: -197.2104340, 454.8011780, -178.9246216, 412.1306152, -609.3410034, 633.7257080
3: -327.8020630, 537.2023926, -297.5463562, 488.1541748, -815.9561157, 834.7487793
4: -286.4281006, 518.3792725, -258.6048279, 471.5862732, -758.0142212, 776.9841309

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9701442, upper bound: 380.9728567
time: 0.96 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9708310, upper bound: 380.9727019
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -120.2597122, 307.3631287, -110.9768829, 284.3798218, -404.6395264, 418.3400269
1: -303.5092163, 464.6740723, -280.7096558, 430.2882690, -733.7974854, 745.3837280
2: -197.2104340, 454.8011780, -181.4240723, 420.3488464, -617.5590820, 636.2252197
3: -327.8020630, 537.2023926, -303.1140747, 497.4658508, -825.2678833, 840.3164673
4: -286.4281006, 518.3792725, -264.0802917, 480.2828064, -766.7109375, 782.4594116

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9701442, upper bound: 380.9728567
time: 0.98 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9708310, upper bound: 380.9727019
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -115.3242798, 295.0093689, -130.1763153, 333.6618652, -448.9861145, 425.1856689
1: -291.9190979, 446.0539551, -329.9284668, 503.3043518, -795.2234497, 775.9824219
2: -189.4988861, 437.3072205, -213.6962128, 494.4724426, -683.9713135, 651.0034180
3: -314.0631104, 516.4781494, -355.0840149, 583.1473389, -897.2103882, 871.5621338
4: -274.6662903, 498.6013794, -308.9260864, 564.0195312, -838.6857910, 807.5274658

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_B1_A1_A1_B1

### Relational analysis result of NS_A2_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9710981, upper bound: 380.9706814
time: 0.70 seconds

## Relational analysis of NS_A2_B2_B1_A1_A1_B2

### Relational analysis result of NS_A2_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9709629, upper bound: 380.9705854
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -119.2465210, 304.2323608, -130.1763153, 333.6618652, -452.9083862, 434.4086609
1: -300.7838135, 459.8878784, -329.9284668, 503.3043518, -804.0881348, 789.8162842
2: -195.6198730, 450.2618408, -213.6962128, 494.4724426, -690.0922852, 663.9580688
3: -324.8850403, 531.8576660, -355.0840149, 583.1473389, -908.0323486, 886.9416504
4: -284.0748291, 512.9515381, -308.9260864, 564.0195312, -848.0942383, 821.8775635

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_B1_A1_A2_B1

### Relational analysis result of NS_A2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9710981, upper bound: 380.9706919
time: 0.72 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2_B2

### Relational analysis result of NS_A2_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9709629, upper bound: 380.9705951
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -144.8809357, 368.3992004, -130.1763153, 333.6618652, -478.5426941, 498.5755005
1: -364.3215027, 555.0853882, -329.9284668, 503.3043518, -867.6258545, 885.0138550
2: -238.4732056, 545.1742554, -213.6962128, 494.4724426, -732.9456787, 758.8704834
3: -392.2914124, 643.4653320, -355.0840149, 583.1473389, -975.4386597, 998.5493164
4: -345.0632019, 620.8609009, -308.9260864, 564.0195312, -909.0827637, 929.7869873

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_B1_A2_A1_B1

### Relational analysis result of NS_A2_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9697433, upper bound: 380.9698916
time: 0.71 seconds

## Relational analysis of NS_A2_B2_B1_A2_A1_B2

### Relational analysis result of NS_A2_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9696036, upper bound: 380.9696970
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -151.6566620, 384.3088379, -130.1763153, 333.6618652, -485.3185425, 514.4851685
1: -380.4439392, 578.4920654, -329.9284668, 503.3043518, -883.7482910, 908.4205322
2: -249.4860382, 568.0228882, -213.6962128, 494.4724426, -743.9584961, 781.7191162
3: -410.5044556, 670.2354736, -355.0840149, 583.1473389, -993.6517944, 1025.3193359
4: -361.0760193, 646.7352905, -308.9260864, 564.0195312, -925.0955811, 955.6613770

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_B1_A2_A2_B1

### Relational analysis result of NS_A2_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9697433, upper bound: 380.9704940
time: 1.04 seconds

## Relational analysis of NS_A2_B2_B1_A2_A2_B2

### Relational analysis result of NS_A2_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9696036, upper bound: 380.9702898
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -115.3253708, 295.0121765, -134.1880035, 343.4147034, -458.7400513, 429.2001343
1: -291.9217224, 446.0583191, -338.8255920, 517.9617310, -809.8834229, 784.8839111
2: -189.5006256, 437.3112793, -219.7531738, 508.3778687, -697.8784790, 657.0644531
3: -314.0659790, 516.4830933, -365.7257385, 599.4548950, -913.5207520, 882.2088013
4: -274.6689453, 498.6060181, -318.5069580, 579.2708130, -853.9397583, 817.1127930

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B2_A1_A1_B1

### Relational analysis result of NS_A2_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9703160, upper bound: 380.9689293
time: 0.72 seconds

## Relational analysis of NS_A2_B2_B2_A1_A1_B2

### Relational analysis result of NS_A2_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9703674, upper bound: 380.9699364
time: 1.14 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -144.8809357, 368.3992004, -134.1880035, 343.4147034, -488.2955627, 502.5871582
1: -364.3215027, 555.0853882, -338.8255920, 517.9617310, -882.2832031, 893.9108887
2: -238.4732056, 545.1742554, -219.7531738, 508.3778687, -746.8510742, 764.9274292
3: -392.2914124, 643.4653320, -365.7257385, 599.4548950, -991.7462769, 1009.1910400
4: -345.0632019, 620.8609009, -318.5069580, 579.2708130, -924.3339844, 939.3677979

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_B2_A1_A2_B1

### Relational analysis result of NS_A2_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9704753, upper bound: 380.9697978
time: 1.18 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2_B2

### Relational analysis result of NS_A2_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9702898, upper bound: 380.9696036
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -119.3278961, 304.4507141, -134.1880035, 343.4147034, -462.7425537, 438.6386719
1: -301.0104065, 460.2100220, -338.8255920, 517.9617310, -818.9720459, 799.0356445
2: -195.7520447, 450.5837708, -219.7531738, 508.3778687, -704.1298828, 670.3369141
3: -325.1102600, 532.2303467, -365.7257385, 599.4548950, -924.5651245, 897.9560547
4: -284.2627869, 513.3238525, -318.5069580, 579.2708130, -863.5335083, 831.8306274

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_B2_A2_A1_A1

### Relational analysis result of NS_A2_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9629021, upper bound: 380.9630060
time: 0.75 seconds

## Relational analysis of NS_A2_B2_B2_A2_A1_A2

### Relational analysis result of NS_A2_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9626729, upper bound: 380.9626729
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -151.6566620, 384.3088379, -134.1880035, 343.4147034, -495.0713501, 518.4968262
1: -380.4439392, 578.4920654, -338.8255920, 517.9617310, -898.4056396, 917.3176270
2: -249.4860382, 568.0228882, -219.7531738, 508.3778687, -757.8638916, 787.7760620
3: -410.5044556, 670.2354736, -365.7257385, 599.4548950, -1009.9593506, 1035.9611816
4: -361.0760193, 646.7352905, -318.5069580, 579.2708130, -940.3468018, 965.2421875

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B2_A2_A2_A1

### Relational analysis result of NS_A2_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9692891, upper bound: 380.9703132
time: 0.70 seconds

## Relational analysis of NS_A2_B2_B2_A2_A2_A2

### Relational analysis result of NS_A2_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9699364, upper bound: 380.9703414
time: 0.70 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.23 seconds
NS_A1_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9763926, upper bound: 380.9763926
NS_A1_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9763926, upper bound: 380.9764540
NS_A1_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9764540, upper bound: 380.9763926
NS_A1_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9764540, upper bound: 380.9764540
NS_A1_B1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9735816, upper bound: 380.9726927
NS_A1_B1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9722911, upper bound: 380.9720856
NS_A1_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9763964, upper bound: 380.9771490
NS_A1_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9763964, upper bound: 380.9771490
NS_A1_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9762363, upper bound: 380.9763048
NS_A1_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9762363, upper bound: 380.9763545
NS_A1_B1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9743737, upper bound: 380.9728527
NS_A1_B1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9763482, upper bound: 380.9763548
NS_A1_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9762453, upper bound: 380.9769157
NS_A1_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9762453, upper bound: 380.9769157
NS_A1_B1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9679076, upper bound: 380.9692741
NS_A1_B1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9762967, upper bound: 380.9768545
NS_A2_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9707196, upper bound: 380.9724326
NS_A2_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9707196, upper bound: 380.9724326
NS_A2_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9705854, upper bound: 380.9721847
NS_A2_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9705854, upper bound: 380.9721847
NS_A2_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9707392, upper bound: 380.9712627
NS_A2_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9707392, upper bound: 380.9712627
NS_A2_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9705951, upper bound: 380.9709629
NS_A2_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9705951, upper bound: 380.9709629
NS_A2_B1_A2_A1_A1_A1, status: Status.VERIFIED, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9563485, upper bound: 380.9524280
NS_A2_B1_A2_A1_A1_A2, status: Status.VERIFIED, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9597056, upper bound: 380.9604470
NS_A2_B1_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9630773, upper bound: 380.9654295
NS_A2_B1_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9629280, upper bound: 380.9650371
NS_A2_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9701442, upper bound: 380.9728567
NS_A2_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9708310, upper bound: 380.9727019
NS_A2_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9701442, upper bound: 380.9728567
NS_A2_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9708310, upper bound: 380.9727019
NS_A2_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9710981, upper bound: 380.9706814
NS_A2_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9709629, upper bound: 380.9705854
NS_A2_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9710981, upper bound: 380.9706919
NS_A2_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9709629, upper bound: 380.9705951
NS_A2_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9697433, upper bound: 380.9698916
NS_A2_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9696036, upper bound: 380.9696970
NS_A2_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9697433, upper bound: 380.9704940
NS_A2_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9696036, upper bound: 380.9702898
NS_A2_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9703160, upper bound: 380.9689293
NS_A2_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9703674, upper bound: 380.9699364
NS_A2_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9704753, upper bound: 380.9697978
NS_A2_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9702898, upper bound: 380.9696036
NS_A2_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9629021, upper bound: 380.9630060
NS_A2_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9626729, upper bound: 380.9626729
NS_A2_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9692891, upper bound: 380.9703132
NS_A2_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -380.9699364, upper bound: 380.9703414

## BFS NS instance: NS_A1_B1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -100.9832230, 259.1163940, -100.9832230, 259.1163940, -360.0995789, 360.0995789
1: -257.3857727, 391.7742920, -257.3857727, 391.7742920, -649.1600342, 649.1600342
2: -166.2153931, 383.3316650, -166.2153931, 383.3316650, -549.5469971, 549.5470581
3: -276.1808472, 453.5200500, -276.1808472, 453.5200500, -729.7006226, 729.7006226
4: -240.2730255, 438.6277466, -240.2730255, 438.6277466, -678.9007568, 678.9007568

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A1_B1_A1_A1

### Relational analysis result of NS_A1_B1_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9739222, upper bound: 380.9739161
time: 0.62 seconds

## Relational analysis of NS_A1_B1_B1_A1_B1_A1_A2

### Relational analysis result of NS_A1_B1_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9750009, upper bound: 380.9750009
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -105.0967255, 269.1310425, -100.9832230, 259.1163940, -364.2131348, 370.1142273
1: -266.2148132, 406.8220520, -257.3857727, 391.7742920, -657.9891357, 664.2078247
2: -172.1022644, 397.4872437, -166.2153931, 383.3316650, -555.4339600, 563.7025757
3: -287.3706360, 470.2888489, -276.1808472, 453.5200500, -740.8906860, 746.4694824
4: -250.3684387, 454.1274109, -240.2730255, 438.6277466, -688.9961548, 694.4004517

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9739160, upper bound: 380.9739942
time: 0.80 seconds

## Relational analysis of NS_A1_B1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9750009, upper bound: 380.9750687
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -100.9832230, 259.1163940, -105.0967255, 269.1310425, -370.1142273, 364.2131348
1: -257.3857727, 391.7742920, -266.2148132, 406.8220520, -664.2078247, 657.9891357
2: -166.2153931, 383.3316650, -172.1022644, 397.4872437, -563.7025757, 555.4339600
3: -276.1808472, 453.5200500, -287.3706360, 470.2888489, -746.4694824, 740.8906860
4: -240.2730255, 438.6277466, -250.3684387, 454.1274109, -694.4004517, 688.9961548

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A1_B2_A1_A1

### Relational analysis result of NS_A1_B1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9739222, upper bound: 380.9739160
time: 0.62 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2_A1_A2

### Relational analysis result of NS_A1_B1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9750009, upper bound: 380.9750009
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -105.0967255, 269.1310425, -105.0967255, 269.1310425, -374.2277527, 374.2277527
1: -266.2148132, 406.8220520, -266.2148132, 406.8220520, -673.0368652, 673.0368652
2: -172.1022644, 397.4872437, -172.1022644, 397.4872437, -569.5894775, 569.5894775
3: -287.3706360, 470.2888489, -287.3706360, 470.2888489, -757.6594849, 757.6594849
4: -250.3684387, 454.1274109, -250.3684387, 454.1274109, -704.4957886, 704.4957886

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A1_B2_A2_A1

### Relational analysis result of NS_A1_B1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9739222, upper bound: 380.9739400
time: 0.64 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2_A2_A2

### Relational analysis result of NS_A1_B1_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9750009, upper bound: 380.9750707
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A1_A1

### Backsubstitution after applying NS history:
0: -104.1331482, 266.4158936, -101.6094666, 260.1312256, -364.2643433, 368.0253601
1: -265.0385437, 403.2050781, -257.2009277, 393.2146912, -658.2532349, 660.4060059
2: -171.0169220, 393.7367859, -166.2576447, 384.2841187, -555.3010254, 559.9944458
3: -285.0068970, 466.2824707, -277.6497803, 454.6197815, -739.6265869, 743.9321899
4: -247.4178162, 450.6841736, -242.0007477, 438.8197327, -686.2374878, 692.6849365

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B1_A2_A1_A1_B1

### Relational analysis result of NS_A1_B1_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9727333, upper bound: 380.9716693
time: 0.65 seconds

## Relational analysis of NS_A1_B1_B1_A2_A1_A1_B2

### Relational analysis result of NS_A1_B1_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9727333, upper bound: 380.9726401
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -106.6542969, 273.1838989, -102.2518311, 262.1799011, -368.8341675, 375.4357300
1: -271.7444153, 413.3524780, -259.1461487, 396.2932434, -668.0376587, 672.4986572
2: -175.3026733, 403.9365234, -167.5470886, 387.4726562, -562.7752686, 571.4835815
3: -291.9770813, 478.4190369, -279.5071411, 458.2772217, -750.2542725, 757.9261475
4: -253.4260712, 462.2277527, -243.6058350, 442.5102234, -695.9362183, 705.8335571

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B1_A2_A1_A2_B1

### Relational analysis result of NS_A1_B1_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9722174, upper bound: 380.9720856
time: 0.66 seconds

## Relational analysis of NS_A1_B1_B1_A2_A1_A2_B2

### Relational analysis result of NS_A1_B1_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9722174, upper bound: 380.9720856
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -111.9040985, 286.3507996, -100.9832230, 259.1163940, -371.0205078, 387.3340149
1: -283.2727661, 433.1998596, -257.3857727, 391.7742920, -675.0470581, 690.5856323
2: -182.9956360, 423.0840759, -166.2153931, 383.3316650, -566.3272705, 589.2993774
3: -305.9629822, 500.5824890, -276.1808472, 453.5200500, -759.4828491, 776.7630615
4: -266.4358215, 483.3019714, -240.2730255, 438.6277466, -705.0635986, 723.5749512

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_B1_A2_A2_B1_B1

### Relational analysis result of NS_A1_B1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9744673, upper bound: 380.9760450
time: 0.90 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B1_A2_A2_B1_B1

### Relational analysis result of NS_A1_B1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9727944, upper bound: 380.9739211
time: 0.71 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B1_A2_A2_B1_A1

### Relational analysis result of NS_A1_B1_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9744000, upper bound: 380.9753705
time: 0.64 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2_B1_A2

### Relational analysis result of NS_A1_B1_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9744000, upper bound: 380.9771490
time: 1.21 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -111.9040985, 286.3507996, -105.0967255, 269.1310425, -381.0351257, 391.4475098
1: -283.2727661, 433.1998596, -266.2148132, 406.8220520, -690.0948486, 699.4146729
2: -182.9956360, 423.0840759, -172.1022644, 397.4872437, -580.4828491, 595.1862793
3: -305.9629822, 500.5824890, -287.3706360, 470.2888489, -776.2516479, 787.9531250
4: -266.4358215, 483.3019714, -250.3684387, 454.1274109, -720.5632324, 733.6703491

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_B1_A2_A2_B2_B1

### Relational analysis result of NS_A1_B1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9744673, upper bound: 380.9761636
time: 0.77 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A2_A2_B2_A1

### Relational analysis result of NS_A1_B1_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9743514, upper bound: 380.9754525
time: 0.68 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2_B2_A2

### Relational analysis result of NS_A1_B1_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9750011, upper bound: 380.9756384
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -100.9832230, 259.1163940, -121.8744736, 310.4247131, -411.4078979, 380.9907532
1: -257.3857727, 391.7742920, -306.7044678, 468.8989563, -726.2846680, 698.4786987
2: -166.2153931, 383.3316650, -200.2115784, 458.5315552, -624.7469482, 583.5432129
3: -276.1808472, 453.5200500, -330.4346008, 542.6766357, -818.8572998, 783.9545288
4: -240.2730255, 438.6277466, -291.0246887, 523.5110474, -763.7840576, 729.6524658

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B2_A1_B1_A1_A1

### Relational analysis result of NS_A1_B1_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9737371, upper bound: 380.9738532
time: 0.66 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_A1_A2

### Relational analysis result of NS_A1_B1_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9747957, upper bound: 380.9749398
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -105.0967255, 269.1310425, -121.8744736, 310.4247131, -415.5214233, 391.0053711
1: -266.2148132, 406.8220520, -306.7044678, 468.8989563, -735.1137695, 713.5264893
2: -172.1022644, 397.4872437, -200.2115784, 458.5315552, -630.6337891, 597.6988525
3: -287.3706360, 470.2888489, -330.4346008, 542.6766357, -830.0472412, 800.7233276
4: -250.3684387, 454.1274109, -291.0246887, 523.5110474, -773.8794556, 745.1520996

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B2_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9737371, upper bound: 380.9738739
time: 0.69 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9747957, upper bound: 380.9749976
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -102.3629913, 262.3944702, -123.9890289, 313.9236145, -416.2866211, 386.3834839
1: -259.2637024, 396.7227783, -311.1682739, 473.5770569, -732.8407593, 707.8910522
2: -167.5481110, 387.6076660, -204.0355682, 463.3184204, -630.8665161, 591.6431274
3: -279.8685913, 458.6397705, -335.9771423, 547.9712524, -827.8397827, 794.6168823
4: -243.8436737, 442.9225159, -295.9992981, 528.5102539, -772.3537598, 738.9218140

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_B2_A1_B2_B1_B1

### Relational analysis result of NS_A1_B1_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9679076, upper bound: 380.9693386
time: 0.69 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_B1_B2

### Relational analysis result of NS_A1_B1_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9731785, upper bound: 380.9723721
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -105.0488892, 269.0461426, -127.0428085, 322.4689331, -427.5178223, 396.0889587
1: -266.1143494, 406.6551514, -319.0991821, 486.6375427, -752.7518921, 725.7543335
2: -172.0497284, 397.3650208, -208.7992706, 475.6862183, -647.7359619, 606.1643066
3: -287.1943359, 470.1423035, -344.6470032, 562.8945312, -850.0888672, 814.7891235
4: -250.2582397, 453.9502869, -303.3480835, 542.8365479, -793.0947876, 757.2982178

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9762363, upper bound: 380.9763048
time: 0.67 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_A2

### Relational analysis result of NS_A1_B1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9762363, upper bound: 380.9763548
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -109.7967453, 280.7104187, -121.8744736, 310.4247131, -420.2214355, 402.5847778
1: -279.6531677, 424.5692444, -306.7044678, 468.8989563, -748.5521240, 731.2736816
2: -180.5979156, 415.0063171, -200.2115784, 458.5315552, -639.1293945, 615.2178955
3: -300.4217224, 491.3782959, -330.4346008, 542.6766357, -843.0983887, 821.8128662
4: -261.0100403, 474.8264465, -291.0246887, 523.5110474, -784.5211182, 765.8511353

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9748313, upper bound: 380.9755334
time: 0.71 seconds

## Relational analysis of NS_A1_B1_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B2_A2_B1_A1_A1

### Relational analysis result of NS_A1_B1_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9742104, upper bound: 380.9750522
time: 0.69 seconds

## Relational analysis of NS_A1_B1_B2_A2_B1_A1_A2

### Relational analysis result of NS_A1_B1_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9747957, upper bound: 380.9754586
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -111.9040985, 286.3507996, -121.8744736, 310.4247131, -422.3287964, 408.2251587
1: -283.2727661, 433.1998596, -306.7044678, 468.8989563, -752.1716919, 739.9042969
2: -182.9956360, 423.0840759, -200.2115784, 458.5315552, -641.5272217, 623.2955933
3: -305.9629822, 500.5824890, -330.4346008, 542.6766357, -848.6395264, 831.0169067
4: -266.4358215, 483.3019714, -291.0246887, 523.5110474, -789.9468384, 774.3266602

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9748313, upper bound: 380.9755334
time: 0.70 seconds

## Relational analysis of NS_A1_B1_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B2_A2_B1_A2_A1

### Relational analysis result of NS_A1_B1_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9742104, upper bound: 380.9750522
time: 0.65 seconds

## Relational analysis of NS_A1_B1_B2_A2_B1_A2_A2

### Relational analysis result of NS_A1_B1_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9747957, upper bound: 380.9754586
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -109.0407867, 279.3946838, -123.3311844, 313.2861633, -422.3269348, 402.7258606
1: -276.0509338, 422.7574158, -310.2349548, 472.4700928, -748.5207520, 732.9923096
2: -178.3049774, 412.6239929, -202.9906311, 462.0647583, -640.3696289, 615.6145630
3: -298.2412415, 488.5007324, -334.9934692, 546.5891113, -844.8303223, 823.4942017
4: -259.6159363, 471.5350342, -294.1887207, 527.4708862, -787.0867920, 765.7237549

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_B2_A2_B2_B1_B1

### Relational analysis result of NS_A1_B1_B2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9679076, upper bound: 380.9692741
time: 0.71 seconds

## Relational analysis of NS_A1_B1_B2_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_B2_A2_B2_B1_B1

### Relational analysis result of NS_A1_B1_B2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9677916, upper bound: 380.9677570
time: 0.78 seconds

## Relational analysis of NS_A1_B1_B2_A2_B2_B1_B2

### Relational analysis result of NS_A1_B1_B2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9678974, upper bound: 380.9692741
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -111.8338165, 286.2111816, -127.0407333, 322.5118408, -434.3456421, 413.2519226
1: -283.1517029, 432.9577942, -319.0720825, 486.7873535, -769.9390259, 752.0299072
2: -182.8950806, 422.8998413, -208.8064728, 475.7397766, -658.6347656, 631.7062378
3: -305.8031921, 500.3356018, -344.7011414, 563.0286255, -868.8317871, 845.0365601
4: -266.1900330, 483.0746765, -303.3609924, 542.9344482, -809.1243896, 786.4356079

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_B2_A2_B2_B2_B1

### Relational analysis result of NS_A1_B1_B2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9731785, upper bound: 380.9727893
time: 0.67 seconds

## Relational analysis of NS_A1_B1_B2_A2_B2_B2_B2

### Relational analysis result of NS_A1_B1_B2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9762967, upper bound: 380.9768545
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -115.3253708, 295.0121765, -109.4662857, 280.2384338, -395.5637512, 404.4784546
1: -291.9217224, 446.0583191, -278.6857300, 423.9806824, -715.9024048, 724.7440186
2: -189.5006256, 437.3112793, -179.9723053, 414.4336548, -603.9342651, 617.2835693
3: -314.0659790, 516.4830933, -299.3610840, 490.8939514, -804.9597778, 815.8441772
4: -274.6689453, 498.6060181, -260.1336975, 474.2632751, -748.9321899, 758.7396240

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B1_B1_A1_A1

### Relational analysis result of NS_A2_B1_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9707196, upper bound: 380.9724326
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B1_B1_A1_A2

### Relational analysis result of NS_A2_B1_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9707196, upper bound: 380.9724326
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -144.8809357, 368.3992004, -109.4662857, 280.2384338, -425.1193237, 477.8654785
1: -364.3215027, 555.0853882, -278.6857300, 423.9806824, -788.3021851, 833.7710571
2: -238.4732056, 545.1742554, -179.9723053, 414.4336548, -652.9068604, 725.1465454
3: -392.2914124, 643.4653320, -299.3610840, 490.8939514, -883.1853027, 942.8264160
4: -345.0632019, 620.8609009, -260.1336975, 474.2632751, -819.3264771, 880.9946289

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B1_B1_A2_A1

### Relational analysis result of NS_A2_B1_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9707196, upper bound: 380.9724326
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A1_B1_B1_A2_A2

### Relational analysis result of NS_A2_B1_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9707196, upper bound: 380.9724326
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -115.3253708, 295.0121765, -109.7886734, 281.0440674, -396.3693848, 404.8008118
1: -291.9217224, 446.0583191, -279.5932922, 425.1916199, -717.1132202, 725.6516113
2: -189.5006256, 437.3112793, -180.5375671, 415.6315613, -605.1321411, 617.8488770
3: -314.0659790, 516.4830933, -300.2929993, 492.3227844, -806.3887329, 816.7760010
4: -274.6689453, 498.6060181, -260.8638306, 475.6730042, -750.3419189, 759.4697876

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B1_B2_A1_A1

### Relational analysis result of NS_A2_B1_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9705854, upper bound: 380.9721847
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B1_B2_A1_A2

### Relational analysis result of NS_A2_B1_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9705854, upper bound: 380.9721847
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -144.8809357, 368.3992004, -109.7886734, 281.0440674, -425.9249268, 478.1878357
1: -364.3215027, 555.0853882, -279.5932922, 425.1916199, -789.5130005, 834.6787109
2: -238.4732056, 545.1742554, -180.5375671, 415.6315613, -654.1047363, 725.7117920
3: -392.2914124, 643.4653320, -300.2929993, 492.3227844, -884.6141968, 943.7583008
4: -345.0632019, 620.8609009, -260.8638306, 475.6730042, -820.7362061, 881.7247314

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B1_B2_A2_A1

### Relational analysis result of NS_A2_B1_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9705854, upper bound: 380.9721847
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A1_B1_B2_A2_A2

### Relational analysis result of NS_A2_B1_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9705854, upper bound: 380.9721847
time: 0.59 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -115.3253708, 295.0121765, -111.4585800, 285.5787048, -400.9040222, 406.4707642
1: -291.9217224, 446.0583191, -281.9114075, 432.1082764, -724.0300293, 727.9697266
2: -189.5006256, 437.3112793, -182.2229309, 422.0971375, -611.5977783, 619.5341797
3: -314.0659790, 516.4830933, -304.4627991, 499.5685120, -813.6342773, 820.9458008
4: -274.6689453, 498.6060181, -265.2369995, 482.2981567, -756.9671021, 763.8429565

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9707392, upper bound: 380.9712627
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A1_B2_B1_A1_A2

### Relational analysis result of NS_A2_B1_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9707392, upper bound: 380.9712627
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -144.8809357, 368.3992004, -111.4585800, 285.5787048, -430.4595337, 479.8577881
1: -364.3215027, 555.0853882, -281.9114075, 432.1082764, -796.4298096, 836.9967041
2: -238.4732056, 545.1742554, -182.2229309, 422.0971375, -660.5703125, 727.3972168
3: -392.2914124, 643.4653320, -304.4627991, 499.5685120, -891.8598022, 947.9281006
4: -345.0632019, 620.8609009, -265.2369995, 482.2981567, -827.3613281, 886.0979004

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9707392, upper bound: 380.9712627
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A1_B2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9707392, upper bound: 380.9712627
time: 0.62 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -115.3253708, 295.0121765, -111.7214584, 286.2406921, -401.5660095, 406.7336426
1: -291.9217224, 446.0583191, -282.6607971, 433.1039124, -725.0256348, 728.7191162
2: -189.5006256, 437.3112793, -182.6893921, 423.0865479, -612.5871582, 620.0006714
3: -314.0659790, 516.4830933, -305.2257690, 500.7460022, -814.8118286, 821.7086182
4: -274.6689453, 498.6060181, -265.8354797, 483.4584351, -758.1273804, 764.4413452

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B2_B2_A1_A1

### Relational analysis result of NS_A2_B1_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9705951, upper bound: 380.9709629
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2_A1_A2

### Relational analysis result of NS_A2_B1_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9705951, upper bound: 380.9709629
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -144.8809357, 368.3992004, -111.7214584, 286.2406921, -431.1215820, 480.1206665
1: -364.3215027, 555.0853882, -282.6607971, 433.1039124, -797.4254150, 837.7462158
2: -238.4732056, 545.1742554, -182.6893921, 423.0865479, -661.5597534, 727.8636475
3: -392.2914124, 643.4653320, -305.2257690, 500.7460022, -893.0373535, 948.6909790
4: -345.0632019, 620.8609009, -265.8354797, 483.4584351, -828.5216064, 886.6963501

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B2_B2_A2_A1

### Relational analysis result of NS_A2_B1_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9705951, upper bound: 380.9709629
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2_A2_A2

### Relational analysis result of NS_A2_B1_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9705951, upper bound: 380.9709629
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A2_A1

### Backsubstitution after applying NS history:
0: -110.8419495, 282.7982788, -106.3657990, 272.8694153, -383.7113647, 389.1640625
1: -279.3635559, 427.9317932, -268.7649231, 413.1755371, -692.5390625, 696.6967163
2: -181.8442230, 418.7502747, -173.6340332, 403.2776184, -585.1218262, 592.3842773
3: -301.9231262, 494.4955139, -290.5021057, 477.5908203, -779.5139160, 784.9976196
4: -263.8953247, 477.0741577, -253.0648346, 460.9141541, -724.8094482, 730.1389160

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_A1_A2_A1_B1

### Relational analysis result of NS_A2_B1_A2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9629280, upper bound: 380.9650371
time: 0.92 seconds

## Relational analysis of NS_A2_B1_A2_A1_A2_A1_B2

### Relational analysis result of NS_A2_B1_A2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9629280, upper bound: 380.9650371
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A2_A2

### Backsubstitution after applying NS history:
0: -112.1033173, 286.5650330, -106.4753647, 273.3925781, -385.4958191, 393.0404053
1: -282.6796265, 433.2838440, -269.2806396, 413.8802795, -696.5599365, 702.5644531
2: -183.8999023, 424.4201965, -173.9187317, 404.1251831, -588.0250244, 598.3389282
3: -305.3132629, 501.0137939, -290.9094238, 478.5240173, -783.8371582, 791.9232178
4: -266.8470764, 483.4403076, -253.2972107, 461.9140015, -728.7609253, 736.7375488

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_A1_A2_A2_B1

### Relational analysis result of NS_A2_B1_A2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9629280, upper bound: 380.9650371
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_A1_A2_A2_B2

### Relational analysis result of NS_A2_B1_A2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9629280, upper bound: 380.9650371
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -113.3088760, 287.8764038, -104.7374878, 268.3704529, -381.6793213, 392.6138611
1: -285.3506470, 434.3605347, -266.5813599, 406.1789551, -691.5296021, 700.9417725
2: -186.3883820, 425.7080078, -172.1266479, 396.5667725, -582.9550781, 597.8346558
3: -308.3400879, 502.3110962, -286.4133606, 470.2449951, -778.5850830, 788.7244873
4: -269.9474182, 484.9534607, -248.9989471, 454.1641846, -724.1114502, 733.9522705

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9699004, upper bound: 380.9725651
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9697750, upper bound: 380.9723069
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -119.3371353, 305.0061035, -107.9568405, 276.4559631, -395.7930908, 412.9629517
1: -301.1253052, 461.1936951, -274.8205872, 418.3891907, -719.5145264, 736.0142822
2: -195.6842957, 451.3133545, -177.4960175, 408.9181213, -604.6024170, 628.8093262
3: -325.2368469, 533.1783447, -295.1608582, 484.4364014, -809.6732178, 828.3390503
4: -284.2623596, 514.4028320, -256.5299683, 467.9347229, -752.1970825, 770.9326782

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9708310, upper bound: 380.9728664
time: 1.35 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9708310, upper bound: 380.9728664
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -113.3088760, 287.8764038, -106.8104477, 273.9580078, -387.2668762, 394.6868591
1: -285.3506470, 434.3605347, -269.9498901, 414.6487732, -699.9993896, 704.3104248
2: -186.3883820, 425.7080078, -174.4843140, 404.6230774, -591.0114136, 600.1923218
3: -308.3400879, 502.3110962, -291.7125549, 479.2556458, -787.5955811, 794.0236206
4: -269.9474182, 484.9534607, -254.3327942, 462.5830688, -732.5302124, 739.2862549

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9695646, upper bound: 380.9691726
time: 1.07 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9695646, upper bound: 380.9727019
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -119.3371353, 305.0061035, -110.1292114, 282.2322388, -401.5693665, 415.1353149
1: -301.1253052, 461.1936951, -278.5535583, 427.1427612, -728.2680664, 739.7472534
2: -195.6842957, 451.3133545, -180.0127411, 417.1768799, -612.8611450, 631.3261108
3: -325.2368469, 533.1783447, -300.8057556, 493.8035583, -819.0403442, 833.9841309
4: -284.2623596, 514.4028320, -262.0785217, 476.6963806, -760.9585571, 776.4812622

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9695646, upper bound: 380.9691726
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9695646, upper bound: 380.9727019
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -115.3242798, 295.0093689, -130.0785675, 333.4157410, -448.7399902, 425.0879517
1: -291.9190979, 446.0539551, -329.6763611, 502.9372559, -794.8563232, 775.7301636
2: -189.4988861, 437.3072205, -213.5310516, 494.1049194, -683.6037598, 650.8382568
3: -314.0631104, 516.4781494, -354.8184204, 582.7195435, -896.7826538, 871.2965698
4: -274.6662903, 498.6013794, -308.6925964, 563.6051636, -838.2714844, 807.2937012

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_B1_A1_A1_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9708540, upper bound: 380.9687133
time: 0.80 seconds

## Relational analysis of NS_A2_B2_B1_A1_A1_B1_A2

### Relational analysis result of NS_A2_B2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9723096, upper bound: 380.9706700
time: 1.17 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -115.3242798, 295.0093689, -130.3821106, 334.1858521, -449.5101318, 425.3914795
1: -291.9190979, 446.0539551, -330.5278625, 504.0882568, -796.0072632, 776.5816650
2: -189.4988861, 437.3072205, -214.0565491, 495.2453308, -684.7442017, 651.3637695
3: -314.0631104, 516.4781494, -355.6995239, 584.0721436, -898.1351929, 872.1776733
4: -274.6662903, 498.6013794, -309.3890991, 564.9411011, -839.6074219, 807.9904175

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_B1_A1_A1_B2_A1

### Relational analysis result of NS_A2_B2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9707498, upper bound: 380.9685350
time: 0.75 seconds

## Relational analysis of NS_A2_B2_B1_A1_A1_B2_A2

### Relational analysis result of NS_A2_B2_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9721847, upper bound: 380.9705731
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -119.2465210, 304.2323608, -130.0785675, 333.4157410, -452.6622620, 434.3109131
1: -300.7838135, 459.8878784, -329.6763611, 502.9372559, -803.7210693, 789.5640869
2: -195.6198730, 450.2618408, -213.5310516, 494.1049194, -689.7247925, 663.7929077
3: -324.8850403, 531.8576660, -354.8184204, 582.7195435, -907.6045532, 886.6760864
4: -284.0748291, 512.9515381, -308.6925964, 563.6051636, -847.6799316, 821.6438599

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_B1_A1_A2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9709629, upper bound: 380.9705951
time: 0.66 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9709629, upper bound: 380.9705951
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -119.2465210, 304.2323608, -130.3821106, 334.1858521, -453.4323730, 434.6144409
1: -300.7838135, 459.8878784, -330.5278625, 504.0882568, -804.8720703, 790.4155273
2: -195.6198730, 450.2618408, -214.0565491, 495.2453308, -690.8652344, 664.3183594
3: -324.8850403, 531.8576660, -355.6995239, 584.0721436, -908.9571533, 887.5571899
4: -284.0748291, 512.9515381, -309.3890991, 564.9411011, -849.0158691, 822.3405762

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_B1_A1_A2_B2_A1

### Relational analysis result of NS_A2_B2_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9709629, upper bound: 380.9705951
time: 0.99 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2_B2_A2

### Relational analysis result of NS_A2_B2_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9709629, upper bound: 380.9705951
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -144.8809357, 368.3992004, -130.0785675, 333.4157410, -478.2965393, 498.4777832
1: -364.3215027, 555.0853882, -329.6763611, 502.9372559, -867.2587280, 884.7616577
2: -238.4732056, 545.1742554, -213.5310516, 494.1049194, -732.5781250, 758.7053223
3: -392.2914124, 643.4653320, -354.8184204, 582.7195435, -975.0109253, 998.2837524
4: -345.0632019, 620.8609009, -308.6925964, 563.6051636, -908.6683350, 929.5533447

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_B1_A2_A1_B1_A1

### Relational analysis result of NS_A2_B2_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9696970, upper bound: 380.9696970
time: 0.80 seconds

## Relational analysis of NS_A2_B2_B1_A2_A1_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9696970, upper bound: 380.9696970
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -144.8809357, 368.3992004, -130.3821106, 334.1858521, -479.0667114, 498.7812805
1: -364.3215027, 555.0853882, -330.5278625, 504.0882568, -868.4097290, 885.6130981
2: -238.4732056, 545.1742554, -214.0565491, 495.2453308, -733.7185059, 759.2308350
3: -392.2914124, 643.4653320, -355.6995239, 584.0721436, -976.3634644, 999.1648560
4: -345.0632019, 620.8609009, -309.3890991, 564.9411011, -910.0042725, 930.2500000

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_B1_A2_A1_B2_A1

### Relational analysis result of NS_A2_B2_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9696970, upper bound: 380.9696970
time: 0.92 seconds

## Relational analysis of NS_A2_B2_B1_A2_A1_B2_A2

### Relational analysis result of NS_A2_B2_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9696970, upper bound: 380.9696970
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -151.6566620, 384.3088379, -130.0785675, 333.4157410, -485.0723877, 514.3873901
1: -380.4439392, 578.4920654, -329.6763611, 502.9372559, -883.3812256, 908.1683350
2: -249.4860382, 568.0228882, -213.5310516, 494.1049194, -743.5908203, 781.5539551
3: -410.5044556, 670.2354736, -354.8184204, 582.7195435, -993.2239990, 1025.0539551
4: -361.0760193, 646.7352905, -308.6925964, 563.6051636, -924.6811523, 955.4277954

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_B1_A2_A2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9696036, upper bound: 380.9702898
time: 0.78 seconds

## Relational analysis of NS_A2_B2_B1_A2_A2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9696036, upper bound: 380.9702898
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -151.6566620, 384.3088379, -130.3821106, 334.1858521, -485.8425293, 514.6909180
1: -380.4439392, 578.4920654, -330.5278625, 504.0882568, -884.5321655, 909.0197754
2: -249.4860382, 568.0228882, -214.0565491, 495.2453308, -744.7313232, 782.0794678
3: -410.5044556, 670.2354736, -355.6995239, 584.0721436, -994.5765381, 1025.9349365
4: -361.0760193, 646.7352905, -309.3890991, 564.9411011, -926.0170898, 956.1243896

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_B1_A2_A2_B2_A1

### Relational analysis result of NS_A2_B2_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9696036, upper bound: 380.9702898
time: 0.67 seconds

## Relational analysis of NS_A2_B2_B1_A2_A2_B2_A2

### Relational analysis result of NS_A2_B2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9696036, upper bound: 380.9702898
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -110.8675613, 283.9117126, -124.9733887, 318.6410217, -429.5085754, 408.8851013
1: -280.6333923, 429.3538208, -314.6173401, 479.7199097, -760.3532715, 743.9710693
2: -182.0880585, 420.7261963, -204.7663269, 471.1581726, -653.2462158, 625.4924927
3: -302.0106201, 497.0350952, -340.1543579, 555.1375732, -857.1481934, 837.1894531
4: -264.0918884, 479.9924622, -296.9103088, 536.6633911, -800.7552490, 776.9026489

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_B2_A1_A1_B1_A1

### Relational analysis result of NS_A2_B2_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9725651, upper bound: 380.9699004
time: 0.66 seconds

## Relational analysis of NS_A2_B2_B2_A1_A1_B1_A2

### Relational analysis result of NS_A2_B2_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9723069, upper bound: 380.9697750
time: 1.00 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -114.5535431, 293.0782776, -133.3146362, 341.0993347, -455.6528931, 426.3929138
1: -289.9089661, 443.2043457, -336.5579834, 514.5271606, -804.4361572, 779.7623291
2: -188.2254181, 434.4450073, -218.3090820, 504.9436340, -693.1690674, 652.7540894
3: -311.9341736, 513.1646729, -363.2647705, 595.4934082, -907.4276123, 876.4294434
4: -272.8391724, 495.3200684, -316.4573669, 575.3444214, -848.1835938, 811.7774048

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_B2_A1_A1_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9725847, upper bound: 380.9706050
time: 0.69 seconds

## Relational analysis of NS_A2_B2_B2_A1_A1_B2_A2

### Relational analysis result of NS_A2_B2_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9723219, upper bound: 380.9704696
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -144.8809357, 368.3992004, -134.0954742, 343.1814270, -488.0622864, 502.4946899
1: -364.3215027, 555.0853882, -338.5868225, 517.6147461, -881.9362183, 893.6721191
2: -238.4732056, 545.1742554, -219.5950470, 508.0306396, -746.5038452, 764.7691650
3: -392.2914124, 643.4653320, -365.4748840, 599.0515137, -991.3428955, 1008.9401855
4: -345.0632019, 620.8609009, -318.2829590, 578.8800049, -923.9432373, 939.1438599

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_B2_A1_A2_B1_A1

### Relational analysis result of NS_A2_B2_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9696970, upper bound: 380.9696036
time: 0.66 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2_B1_A2

### Relational analysis result of NS_A2_B2_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9696970, upper bound: 380.9696036
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -144.8809357, 368.3992004, -134.4060669, 343.9693298, -488.8501587, 502.8052673
1: -364.3215027, 555.0853882, -339.4558716, 518.7958374, -883.1173096, 894.5412598
2: -238.4732056, 545.1742554, -220.1311035, 509.2065735, -747.6798096, 765.3052979
3: -392.2914124, 643.4653320, -366.3619690, 600.4426270, -992.7339478, 1009.8272705
4: -345.0632019, 620.8609009, -318.9950256, 580.2438354, -925.3070068, 939.8558350

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9702898, upper bound: 380.9696036
time: 0.72 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9702898, upper bound: 380.9696036
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_B2_A2_A1_A1

### Backsubstitution after applying NS history:
0: -113.3789444, 289.3088379, -131.3379211, 336.1997070, -449.5786438, 420.6467590
1: -285.6094971, 437.8147278, -331.4428406, 507.1714478, -792.7808838, 769.2575073
2: -185.8128662, 428.2245483, -214.9120941, 497.6008301, -683.4136963, 643.1364746
3: -308.8887634, 506.0771484, -357.9759827, 586.8857422, -895.7745361, 864.0529785
4: -270.0323486, 487.8211365, -311.7683716, 567.0126953, -837.0449219, 799.5894775

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_B2_A2_A1_A1_B1

### Relational analysis result of NS_A2_B2_B2_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9634603, upper bound: 380.9629367
time: 1.03 seconds

## Relational analysis of NS_A2_B2_B2_A2_A1_A1_B2

### Relational analysis result of NS_A2_B2_B2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9634603, upper bound: 380.9629367
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_B2_A2_A1_A2

### Backsubstitution after applying NS history:
0: -116.2811203, 297.1975098, -132.1430511, 338.5377502, -454.8188782, 429.3405762
1: -293.3865356, 449.3153076, -333.7175903, 510.6950378, -804.0815430, 783.0328979
2: -190.6967316, 440.0096130, -216.3668671, 501.2495422, -691.9462891, 656.3764648
3: -316.8699951, 519.6517334, -360.2341003, 591.0341187, -907.9041138, 879.8858643
4: -276.8942261, 501.3011780, -313.4963684, 571.1868896, -848.0810547, 814.7975464

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_B2_A2_A1_A2_B1

### Relational analysis result of NS_A2_B2_B2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9645230, upper bound: 380.9630469
time: 0.87 seconds

## Relational analysis of NS_A2_B2_B2_A2_A1_A2_B2

### Relational analysis result of NS_A2_B2_B2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9645230, upper bound: 380.9630469
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_B2_A2_A2_A1

### Backsubstitution after applying NS history:
0: -144.4506836, 364.0999146, -129.1220093, 330.8854065, -475.3360901, 493.2218933
1: -361.6000977, 546.9607544, -325.7638245, 499.0997314, -860.6998291, 872.7246094
2: -238.0000153, 537.5017700, -211.1646271, 489.5381165, -727.5380859, 748.6663818
3: -390.5221863, 633.9522095, -351.9972534, 577.4317017, -967.9538574, 985.9494629
4: -343.9829102, 612.1051025, -306.6309814, 558.0077515, -901.9906616, 918.7360229

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_B2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_B2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_B2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_B2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_B2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_B2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B2_A2_A2_A1_B1

### Relational analysis result of NS_A2_B2_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9692657, upper bound: 380.9691726
time: 0.72 seconds

## Relational analysis of NS_A2_B2_B2_A2_A2_A1_B2

### Relational analysis result of NS_A2_B2_B2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9692657, upper bound: 380.9703126
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_B2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -150.8312683, 382.0941162, -133.6248779, 341.9227905, -492.7540588, 515.7189331
1: -378.2218628, 575.1597900, -337.3686523, 515.7516479, -893.9733276, 912.5284424
2: -248.1304016, 564.7034302, -218.8254089, 506.1648560, -754.2952271, 783.5288086
3: -408.1328735, 666.3998413, -364.1385803, 596.9065552, -1005.0393066, 1030.5384521
4: -359.1343079, 642.9506226, -317.1835022, 576.7406616, -935.8750000, 960.1340942

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B2_A2_A2_A2_B1

### Relational analysis result of NS_A2_B2_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9693303, upper bound: 380.9691726
time: 1.09 seconds

## Relational analysis of NS_A2_B2_B2_A2_A2_A2_B2

### Relational analysis result of NS_A2_B2_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9693303, upper bound: 380.9703414
time: 0.77 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.54 seconds
NS_A1_B1_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9739222, upper bound: 380.9739161
NS_A1_B1_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9750009, upper bound: 380.9750009
NS_A1_B1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9739160, upper bound: 380.9739942
NS_A1_B1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9750009, upper bound: 380.9750687
NS_A1_B1_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9739222, upper bound: 380.9739160
NS_A1_B1_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9750009, upper bound: 380.9750009
NS_A1_B1_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9739222, upper bound: 380.9739400
NS_A1_B1_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9750009, upper bound: 380.9750707
NS_A1_B1_B1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9727333, upper bound: 380.9716693
NS_A1_B1_B1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9727333, upper bound: 380.9726401
NS_A1_B1_B1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9722174, upper bound: 380.9720856
NS_A1_B1_B1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9722174, upper bound: 380.9720856
NS_A1_B1_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9744000, upper bound: 380.9753705
NS_A1_B1_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9744000, upper bound: 380.9771490
NS_A1_B1_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9743514, upper bound: 380.9754525
NS_A1_B1_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9750011, upper bound: 380.9756384
NS_A1_B1_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9737371, upper bound: 380.9738532
NS_A1_B1_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9747957, upper bound: 380.9749398
NS_A1_B1_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9737371, upper bound: 380.9738739
NS_A1_B1_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9747957, upper bound: 380.9749976
NS_A1_B1_B2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9679076, upper bound: 380.9693386
NS_A1_B1_B2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9731785, upper bound: 380.9723721
NS_A1_B1_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9762363, upper bound: 380.9763048
NS_A1_B1_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9762363, upper bound: 380.9763548
NS_A1_B1_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9742104, upper bound: 380.9750522
NS_A1_B1_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9747957, upper bound: 380.9754586
NS_A1_B1_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9742104, upper bound: 380.9750522
NS_A1_B1_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9747957, upper bound: 380.9754586
NS_A1_B1_B2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9677916, upper bound: 380.9677570
NS_A1_B1_B2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9678974, upper bound: 380.9692741
NS_A1_B1_B2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9731785, upper bound: 380.9727893
NS_A1_B1_B2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9762967, upper bound: 380.9768545
NS_A2_B1_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9707196, upper bound: 380.9724326
NS_A2_B1_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9707196, upper bound: 380.9724326
NS_A2_B1_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9707196, upper bound: 380.9724326
NS_A2_B1_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9707196, upper bound: 380.9724326
NS_A2_B1_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9705854, upper bound: 380.9721847
NS_A2_B1_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9705854, upper bound: 380.9721847
NS_A2_B1_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9705854, upper bound: 380.9721847
NS_A2_B1_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9705854, upper bound: 380.9721847
NS_A2_B1_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9707392, upper bound: 380.9712627
NS_A2_B1_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9707392, upper bound: 380.9712627
NS_A2_B1_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9707392, upper bound: 380.9712627
NS_A2_B1_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9707392, upper bound: 380.9712627
NS_A2_B1_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9705951, upper bound: 380.9709629
NS_A2_B1_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9705951, upper bound: 380.9709629
NS_A2_B1_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9705951, upper bound: 380.9709629
NS_A2_B1_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9705951, upper bound: 380.9709629
NS_A2_B1_A2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9629280, upper bound: 380.9650371
NS_A2_B1_A2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9629280, upper bound: 380.9650371
NS_A2_B1_A2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9629280, upper bound: 380.9650371
NS_A2_B1_A2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9629280, upper bound: 380.9650371
NS_A2_B1_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9699004, upper bound: 380.9725651
NS_A2_B1_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9697750, upper bound: 380.9723069
NS_A2_B1_A2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9708310, upper bound: 380.9728664
NS_A2_B1_A2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9708310, upper bound: 380.9728664
NS_A2_B1_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9695646, upper bound: 380.9691726
NS_A2_B1_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9695646, upper bound: 380.9727019
NS_A2_B1_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9695646, upper bound: 380.9691726
NS_A2_B1_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9695646, upper bound: 380.9727019
NS_A2_B2_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9708540, upper bound: 380.9687133
NS_A2_B2_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9723096, upper bound: 380.9706700
NS_A2_B2_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9707498, upper bound: 380.9685350
NS_A2_B2_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9721847, upper bound: 380.9705731
NS_A2_B2_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9709629, upper bound: 380.9705951
NS_A2_B2_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9709629, upper bound: 380.9705951
NS_A2_B2_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9709629, upper bound: 380.9705951
NS_A2_B2_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9709629, upper bound: 380.9705951
NS_A2_B2_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9696970, upper bound: 380.9696970
NS_A2_B2_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9696970, upper bound: 380.9696970
NS_A2_B2_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9696970, upper bound: 380.9696970
NS_A2_B2_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9696970, upper bound: 380.9696970
NS_A2_B2_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9696036, upper bound: 380.9702898
NS_A2_B2_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9696036, upper bound: 380.9702898
NS_A2_B2_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9696036, upper bound: 380.9702898
NS_A2_B2_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9696036, upper bound: 380.9702898
NS_A2_B2_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9725651, upper bound: 380.9699004
NS_A2_B2_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9723069, upper bound: 380.9697750
NS_A2_B2_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9725847, upper bound: 380.9706050
NS_A2_B2_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9723219, upper bound: 380.9704696
NS_A2_B2_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9696970, upper bound: 380.9696036
NS_A2_B2_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9696970, upper bound: 380.9696036
NS_A2_B2_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9702898, upper bound: 380.9696036
NS_A2_B2_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9702898, upper bound: 380.9696036
NS_A2_B2_B2_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9634603, upper bound: 380.9629367
NS_A2_B2_B2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9634603, upper bound: 380.9629367
NS_A2_B2_B2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9645230, upper bound: 380.9630469
NS_A2_B2_B2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9645230, upper bound: 380.9630469
NS_A2_B2_B2_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9692657, upper bound: 380.9691726
NS_A2_B2_B2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9692657, upper bound: 380.9703126
NS_A2_B2_B2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9693303, upper bound: 380.9691726
NS_A2_B2_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -380.9693303, upper bound: 380.9703414

## BFS NS instance: NS_A1_B1_B1_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -99.7526932, 255.9207916, -100.7998810, 258.6242676, -358.3768921, 356.7206726
1: -254.0900574, 386.9746094, -256.8713989, 391.0328064, -645.1227417, 643.8460083
2: -164.1289368, 378.6371460, -165.8987122, 382.6050110, -546.7337646, 544.5358887
3: -272.7788696, 447.9710693, -275.6653748, 452.6593018, -725.4381104, 723.6363525
4: -237.4080963, 433.1632385, -239.8563385, 437.7664795, -675.1744385, 673.0195312

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_B1_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_B1_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A1_B1_A1_A1_B1

### Relational analysis result of NS_A1_B1_B1_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9728516, upper bound: 380.9728516
time: 0.69 seconds

## Relational analysis of NS_A1_B1_B1_A1_B1_A1_A1_B2

### Relational analysis result of NS_A1_B1_B1_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9728516, upper bound: 380.9739161
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -100.4225082, 257.7239990, -100.8858490, 258.8743896, -359.2969055, 358.6098328
1: -255.9064636, 389.6996155, -257.1289368, 391.4138794, -647.3203125, 646.8285522
2: -165.2433319, 381.2503357, -166.0466003, 382.9699097, -548.2131348, 547.2969360
3: -274.6385498, 451.0851746, -275.9133301, 453.0969238, -727.7354126, 726.9983521
4: -238.9626007, 436.2673340, -240.0456238, 438.2172546, -677.1798706, 676.3128662

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_B1_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_B1_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A1_B1_A1_A2_B1

### Relational analysis result of NS_A1_B1_B1_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9739161, upper bound: 380.9739222
time: 0.71 seconds

## Relational analysis of NS_A1_B1_B1_A1_B1_A1_A2_B2

### Relational analysis result of NS_A1_B1_B1_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9739161, upper bound: 380.9750009
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -104.9235229, 268.6634216, -99.7526932, 255.9207916, -360.8442993, 368.4160461
1: -265.7404480, 406.1138306, -254.0900574, 386.9746094, -652.7150879, 660.2038574
2: -171.8152466, 396.7975464, -164.1289368, 378.6371460, -550.4523926, 560.9263306
3: -286.8858337, 469.4714661, -272.7788696, 447.9710693, -734.8569336, 742.2503052
4: -249.9762726, 453.3120117, -237.4080963, 433.1632385, -683.1395264, 690.7200317

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9728516, upper bound: 380.9728623
time: 0.99 seconds

## Relational analysis of NS_A1_B1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9728516, upper bound: 380.9739942
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -104.9950943, 268.8772583, -100.4225082, 257.7239990, -362.7190857, 369.2997742
1: -265.9385986, 406.4464722, -255.9064636, 389.6996155, -655.6381836, 662.3528442
2: -171.9205933, 397.1069641, -165.2433319, 381.2503357, -553.1708984, 562.3499756
3: -287.0894775, 469.8461304, -274.6385498, 451.0851746, -738.1746826, 744.4846802
4: -250.1321411, 453.6945496, -238.9626007, 436.2673340, -686.3993530, 692.6570435

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9739222, upper bound: 380.9739401
time: 0.73 seconds

## Relational analysis of NS_A1_B1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9739222, upper bound: 380.9750687
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -99.7526932, 255.9207916, -104.9235229, 268.6634216, -368.4160461, 360.8442993
1: -254.0900574, 386.9746094, -265.7404480, 406.1138306, -660.2037964, 652.7150879
2: -164.1289368, 378.6371460, -171.8152466, 396.7975464, -560.9263306, 550.4523926
3: -272.7788696, 447.9710693, -286.8858337, 469.4714661, -742.2503052, 734.8569336
4: -237.4080963, 433.1632385, -249.9762726, 453.3120117, -690.7200317, 683.1395264

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_B1_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_B1_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_B1_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A1_B2_A1_A1_B1

### Relational analysis result of NS_A1_B1_B1_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9728623, upper bound: 380.9728517
time: 0.66 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2_A1_A1_B2

### Relational analysis result of NS_A1_B1_B1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9728623, upper bound: 380.9739160
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -100.4225082, 257.7239990, -104.9950943, 268.8772583, -369.2997742, 362.7190857
1: -255.9064636, 389.6996155, -265.9385986, 406.4464722, -662.3528442, 655.6381836
2: -165.2433319, 381.2503357, -171.9205933, 397.1069641, -562.3499756, 553.1708984
3: -274.6385498, 451.0851746, -287.0894775, 469.8461304, -744.4846802, 738.1746826
4: -238.9626007, 436.2673340, -250.1321411, 453.6945496, -692.6570435, 686.3993530

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_B1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_B1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B1_B1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9739401, upper bound: 380.9739222
time: 0.68 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B1_B1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9739401, upper bound: 380.9750009
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -103.6710358, 265.4996338, -104.9235229, 268.6634216, -372.3344116, 370.4230957
1: -262.5648193, 401.3582764, -265.7404480, 406.1138306, -668.6786499, 667.0987549
2: -169.7539825, 392.2091064, -171.8152466, 396.7975464, -566.5515137, 564.0243530
3: -283.4833374, 464.0074463, -286.8858337, 469.4714661, -752.9548340, 750.8933105
4: -246.9916534, 448.0191650, -249.9762726, 453.3120117, -700.3035889, 697.9953613

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_B1_A1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B1_B1_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9716975, upper bound: 380.9729902
time: 1.02 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B1_B1_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9714459, upper bound: 380.9709912
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -104.4988327, 267.6361694, -104.9950943, 268.8772583, -373.3760986, 372.6312561
1: -264.5894775, 404.6087341, -265.9385986, 406.4464722, -671.0358887, 670.5473633
2: -171.0325623, 395.2520752, -171.9205933, 397.1069641, -568.1391602, 567.1726685
3: -285.7185059, 467.6827393, -287.0894775, 469.8461304, -755.5646362, 754.7722168
4: -248.9765015, 451.5820618, -250.1321411, 453.6945496, -702.6709595, 701.7141724

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_B1_A1_B2_A2_A2_A1

### Relational analysis result of NS_A1_B1_B1_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9724629, upper bound: 380.9721940
time: 1.09 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2_A2_A2_A2

### Relational analysis result of NS_A1_B1_B1_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9722002, upper bound: 380.9721940
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -99.6672745, 255.1684570, -94.4409866, 239.9489899, -339.6162415, 349.6094360
1: -253.4591675, 386.3490906, -238.3238831, 361.7304993, -615.1896973, 624.6729736
2: -163.4832764, 376.7376099, -154.9952087, 354.0971375, -517.5803833, 531.7327881
3: -272.7998962, 446.6638489, -257.6497803, 418.2565613, -691.0564575, 704.3135986
4: -236.9190063, 431.5995483, -225.1187744, 403.9559937, -640.8749390, 656.7182007

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B1_A2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_B1_A2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_B1_A2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_B1_A2_A1_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1_A2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9715395, upper bound: 380.9698920
time: 0.97 seconds

## Relational analysis of NS_A1_B1_B1_A2_A1_A1_B1_B2

### Relational analysis result of NS_A1_B1_B1_A2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9715173, upper bound: 380.9698674
time: 1.21 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -103.4449768, 264.6981812, -101.0824432, 258.7720642, -362.2169800, 365.7806396
1: -263.3011169, 400.6862488, -255.8378906, 391.1724548, -654.4735718, 656.5241699
2: -169.8914490, 391.2252197, -165.3905029, 382.2854309, -552.1768188, 556.6157227
3: -283.1290588, 463.3485718, -276.1666260, 452.2752686, -735.4042969, 739.5151978
4: -245.7827454, 447.8172913, -240.7419586, 436.5420227, -682.3246460, 688.5590210

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_B1_A2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B1_A2_A1_A1_B2_B1

### Relational analysis result of NS_A1_B1_B1_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9732888, upper bound: 380.9726401
time: 0.72 seconds

## Relational analysis of NS_A1_B1_B1_A2_A1_A1_B2_B2

### Relational analysis result of NS_A1_B1_B1_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9732888, upper bound: 380.9726401
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -106.6542969, 273.1838989, -97.9530487, 250.7856750, -357.4399414, 371.1369629
1: -271.7444153, 413.3524780, -247.6056824, 379.3793335, -651.1237793, 660.9581299
2: -175.3026733, 403.9365234, -160.1527100, 370.4370728, -545.7397461, 564.0892334
3: -291.9770813, 478.4190369, -267.5650940, 438.4303589, -730.4074707, 745.9839478
4: -253.4260712, 462.2277527, -233.3290558, 423.0120239, -676.4381104, 695.5567627

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_B1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_B1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_B1_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_B1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_B1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A2_A1_A2_B1_B1

### Relational analysis result of NS_A1_B1_B1_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9689140, upper bound: 380.9690706
time: 0.75 seconds

## Relational analysis of NS_A1_B1_B1_A2_A1_A2_B1_B2

### Relational analysis result of NS_A1_B1_B1_A2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9702548, upper bound: 380.9702068
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -106.6542969, 273.1838989, -100.9649582, 259.0984802, -365.7527466, 374.1488342
1: -271.7444153, 413.3524780, -255.9429169, 391.7223206, -663.4667358, 669.2954102
2: -175.3026733, 403.9365234, -165.4116821, 382.9905396, -558.2931519, 569.3482056
3: -291.9770813, 478.4190369, -276.0328674, 452.9907227, -744.9677734, 754.4519043
4: -253.4260712, 462.2277527, -240.4955750, 437.4316101, -690.8576660, 702.7233276

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_B1_A2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B1_A2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_B1_A2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_B1_A2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_B1_A2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_B1_A2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_B1_A2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A2_A1_A2_B2_B1

### Relational analysis result of NS_A1_B1_B1_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9689140, upper bound: 380.9690706
time: 1.01 seconds

## Relational analysis of NS_A1_B1_B1_A2_A1_A2_B2_B2

### Relational analysis result of NS_A1_B1_B1_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9702548, upper bound: 380.9702068
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -101.9748535, 259.9305420, -96.4044418, 247.6397095, -349.6144714, 356.3349609
1: -257.0855713, 392.5806580, -245.7299042, 374.4460754, -631.5314941, 638.3105469
2: -166.3242798, 383.4334717, -158.6664581, 366.0630493, -532.3873291, 542.0999146
3: -278.6247864, 453.3329773, -263.7961731, 433.3022156, -711.9269409, 717.1291504
4: -242.9478455, 437.4482117, -229.4826355, 419.2309265, -662.1787720, 666.9307861

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_B1_A2_A2_B1_A1_A1

### Relational analysis result of NS_A1_B1_B1_A2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9734851, upper bound: 380.9747987
time: 0.71 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_B1_A2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_B1_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9733000, upper bound: 380.9739264
time: 0.62 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_B1_A2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_B1_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9741909, upper bound: 380.9750933
time: 0.67 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_B1_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9740522, upper bound: 380.9749862
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -110.8502502, 283.6564941, -100.2287979, 257.2172852, -368.0675354, 383.8852539
1: -280.5485535, 429.2242432, -255.4658051, 388.9543152, -669.5028687, 684.6900635
2: -181.2368622, 419.0810547, -164.9706268, 380.5511169, -561.7879639, 584.0515747
3: -303.0766907, 495.9345398, -274.1262817, 450.2586670, -753.3352661, 770.0607910
4: -263.9597473, 478.7846985, -238.4566040, 435.4484558, -699.4082031, 717.2413330

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_B1_A2_A2_B1_A2_A1

### Relational analysis result of NS_A1_B1_B1_A2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9743557, upper bound: 380.9756640
time: 0.71 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2_B1_A2_A2

### Relational analysis result of NS_A1_B1_B1_A2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9754646, upper bound: 380.9757175
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -110.9411240, 283.7304077, -104.9235229, 268.6634216, -379.6044922, 388.6539307
1: -280.5237427, 429.2385559, -265.7404480, 406.1138306, -686.6375732, 694.9790039
2: -181.2866058, 419.1528931, -171.8152466, 396.7975464, -578.0841064, 590.9681396
3: -303.2391663, 496.0000000, -286.8858337, 469.4714661, -772.7105713, 782.8858643
4: -264.2347412, 478.6908264, -249.9762726, 453.3120117, -717.5466309, 728.6671143

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_B1_A2_A2_B2_A1_A1

### Relational analysis result of NS_A1_B1_B1_A2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9715576, upper bound: 380.9711066
time: 0.70 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2_B2_A1_A2

### Relational analysis result of NS_A1_B1_B1_A2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9714459, upper bound: 380.9714423
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -111.4611664, 285.2607117, -104.9950943, 268.8772583, -380.3384399, 390.2557983
1: -282.1170044, 431.5863953, -265.9385986, 406.4464722, -688.5634766, 697.5248413
2: -182.2482147, 421.4548950, -171.9205933, 397.1069641, -579.3550415, 593.3754883
3: -304.7623901, 498.6835632, -287.0894775, 469.8461304, -774.6085205, 785.7730103
4: -265.4147644, 481.4683838, -250.1321411, 453.6945496, -719.1092529, 731.6005249

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_B1_A2_A2_B2_A2_A1

### Relational analysis result of NS_A1_B1_B1_A2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9724629, upper bound: 380.9721940
time: 0.70 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2_B2_A2_A2

### Relational analysis result of NS_A1_B1_B1_A2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9724505, upper bound: 380.9721940
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -99.7526932, 255.9207916, -121.7199631, 309.9952698, -409.7479248, 377.6407471
1: -254.0900574, 386.9746094, -306.2681274, 468.2433777, -722.3334351, 693.2427368
2: -164.1289368, 378.6371460, -199.9503632, 457.8869934, -622.0157471, 578.5875244
3: -272.7788696, 447.9710693, -329.9908447, 541.9181519, -814.6968994, 777.9619141
4: -237.4080963, 433.1632385, -290.6737671, 522.7520142, -760.1600952, 723.8369141

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_B2_A1_B1_A1_A1_A1

### Relational analysis result of NS_A1_B1_B2_A1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9735116, upper bound: 380.9734956
time: 1.17 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_A1_A1_A2

### Relational analysis result of NS_A1_B1_B2_A1_B1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9609241, upper bound: 380.9587638
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -100.4225082, 257.7239990, -121.7696304, 310.1657410, -410.5882263, 379.4936218
1: -255.9064636, 389.6996155, -306.4219055, 468.5155945, -724.4220581, 696.1215210
2: -165.2433319, 381.2503357, -200.0270386, 458.1473083, -623.3905640, 581.2773438
3: -274.6385498, 451.0851746, -330.1451416, 542.2260742, -816.8645630, 781.2302856
4: -238.9626007, 436.2673340, -290.7833557, 523.0679932, -762.0305786, 727.0505981

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B2_A1_B1_A1_A2_B1

### Relational analysis result of NS_A1_B1_B2_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9734547, upper bound: 380.9736215
time: 1.27 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_A1_A2_B2

### Relational analysis result of NS_A1_B1_B2_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9734547, upper bound: 380.9749398
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -103.6710358, 265.4996338, -121.7199631, 309.9952698, -413.6663208, 387.2196045
1: -262.5648193, 401.3582764, -306.2681274, 468.2433777, -730.8082275, 707.6264038
2: -169.7539825, 392.2091064, -199.9503632, 457.8869934, -627.6408691, 592.1594849
3: -283.4833374, 464.0074463, -329.9908447, 541.9181519, -825.4014893, 793.9982910
4: -246.9916534, 448.0191650, -290.6737671, 522.7520142, -769.7436523, 738.6928711

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B2_A1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B1_B2_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9723904, upper bound: 380.9725613
time: 0.74 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B1_B2_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9723904, upper bound: 380.9738739
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -104.4988327, 267.6361694, -121.7696304, 310.1657410, -414.6645813, 389.4057617
1: -264.5894775, 404.6087341, -306.4219055, 468.5155945, -733.1049805, 711.0306396
2: -171.0325623, 395.2520752, -200.0270386, 458.1473083, -629.1797485, 595.2791138
3: -285.7185059, 467.6827393, -330.1451416, 542.2260742, -827.9445801, 797.8278809
4: -248.9765015, 451.5820618, -290.7833557, 523.0679932, -772.0444946, 742.3654175

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B2_A1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_B2_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9734531, upper bound: 380.9736957
time: 1.05 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_B2_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9734531, upper bound: 380.9749976
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -98.6684113, 253.4140320, -119.3649597, 302.3802795, -401.0487061, 372.7789917
1: -249.9887543, 383.2277527, -300.1790466, 455.7098083, -705.6984863, 683.4067993
2: -161.4736786, 374.1772156, -196.8402405, 446.1669006, -607.6405640, 571.0174561
3: -269.9666138, 442.9642639, -323.8636169, 527.5173340, -797.4838867, 766.8278809
4: -234.8803864, 427.8533020, -284.6339722, 509.2366638, -744.1170654, 712.4872437

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_B2_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_B2_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B2_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B2_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_B2_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_B2_A1_B2_B1_B1_B1

### Relational analysis result of NS_A1_B1_B2_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9677916, upper bound: 380.9679773
time: 0.70 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_B1_B1_B2

### Relational analysis result of NS_A1_B1_B2_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9678974, upper bound: 380.9693386
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -101.7879639, 260.9498901, -122.9622726, 311.3305969, -413.1185303, 383.9121704
1: -257.7808228, 394.5690002, -308.4754944, 469.6871033, -727.4678345, 703.0444946
2: -166.5943756, 385.4627686, -202.3161621, 459.4302979, -626.0246582, 587.7789307
3: -278.2857971, 456.1367493, -333.1696472, 543.4403076, -821.7260132, 789.3063354
4: -242.4827118, 440.4846191, -293.5801086, 524.0889893, -766.5717163, 734.0646973

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_B2_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_B2_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B2_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B2_A1_B2_B1_B2_B1

### Relational analysis result of NS_A1_B1_B2_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9720970, upper bound: 380.9717574
time: 0.64 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_B2_A1_B2_B1_B2_B1

### Relational analysis result of NS_A1_B1_B2_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9718286, upper bound: 380.9713722
time: 1.02 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_B1_B2_B2

### Relational analysis result of NS_A1_B1_B2_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9718188, upper bound: 380.9715546
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -100.3701019, 257.5316162, -127.0428085, 322.4689331, -422.8389587, 384.5744324
1: -255.8029175, 389.3551941, -319.0991821, 486.6375427, -742.4404297, 708.4543457
2: -165.2171631, 380.9987488, -208.7992706, 475.6862183, -640.9033813, 589.7980347
3: -274.4467773, 450.7568359, -344.6470032, 562.8945312, -837.3413086, 795.4035034
4: -238.8116608, 435.9245605, -303.3480835, 542.8365479, -781.6480103, 739.2725220

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_A1_A1

### Relational analysis result of NS_A1_B1_B2_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9737371, upper bound: 380.9738532
time: 0.96 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_A1_A2

### Relational analysis result of NS_A1_B1_B2_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9747957, upper bound: 380.9749398
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -104.5839996, 267.8203430, -127.0428085, 322.4689331, -427.0529175, 394.8631592
1: -264.8865967, 404.8282471, -319.0991821, 486.6375427, -751.5240479, 723.9274292
2: -171.2581635, 395.5346680, -208.7992706, 475.6862183, -646.9443359, 604.3339233
3: -285.9227295, 467.9893188, -344.6470032, 562.8945312, -848.8172607, 812.6359863
4: -249.1576080, 451.8865051, -303.3480835, 542.8365479, -791.9941406, 755.2345581

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_A2_A1

### Relational analysis result of NS_A1_B1_B2_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9737371, upper bound: 380.9738739
time: 0.80 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_B2_A2_A2

### Relational analysis result of NS_A1_B1_B2_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9747957, upper bound: 380.9750023
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_B2_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -108.7000504, 277.7716370, -121.7199631, 309.9952698, -418.6953125, 399.4915466
1: -276.5914307, 420.1418457, -306.2681274, 468.2433777, -744.8348389, 726.4099121
2: -178.6707001, 410.6059875, -199.9503632, 457.8869934, -636.5576782, 610.5563354
3: -297.3731079, 486.2483826, -329.9908447, 541.9181519, -839.2912598, 816.2392578
4: -258.5047913, 469.6962585, -290.6737671, 522.7520142, -781.2567749, 760.3699951

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_B2_A2_B1_A1_A1_A1

### Relational analysis result of NS_A1_B1_B2_A2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9739731, upper bound: 380.9748770
time: 0.63 seconds

## Relational analysis of NS_A1_B1_B2_A2_B1_A1_A1_A2

### Relational analysis result of NS_A1_B1_B2_A2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9734539, upper bound: 380.9745398
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_B2_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -109.3353500, 279.5461121, -121.7696304, 310.1657410, -419.5010681, 401.3156738
1: -278.4280090, 422.8410339, -306.4219055, 468.5155945, -746.9436035, 729.2629395
2: -179.8103180, 413.2592773, -200.0270386, 458.1473083, -637.9576416, 613.2863159
3: -299.1432190, 489.3517151, -330.1451416, 542.2260742, -841.3692627, 819.4967041
4: -259.9348145, 472.8533325, -290.7833557, 523.0679932, -783.0027466, 763.6367188

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_B2_A2_B1_A1_A2_A1

### Relational analysis result of NS_A1_B1_B2_A2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9745779, upper bound: 380.9751382
time: 0.73 seconds

## Relational analysis of NS_A1_B1_B2_A2_B1_A1_A2_A2

### Relational analysis result of NS_A1_B1_B2_A2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9743908, upper bound: 380.9750655
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_B2_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -110.9411240, 283.7304077, -121.7199631, 309.9952698, -420.9364014, 405.4503784
1: -280.5237427, 429.2385559, -306.2681274, 468.2433777, -748.7670898, 735.5067139
2: -181.2866058, 419.1528931, -199.9503632, 457.8869934, -639.1734619, 619.1031494
3: -303.2391663, 496.0000000, -329.9908447, 541.9181519, -845.1572266, 825.9908447
4: -264.2347412, 478.6908264, -290.6737671, 522.7520142, -786.9866333, 769.3646240

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B2_A2_B1_A2_A1_B1

### Relational analysis result of NS_A1_B1_B2_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9723904, upper bound: 380.9738531
time: 0.73 seconds

## Relational analysis of NS_A1_B1_B2_A2_B1_A2_A1_B2

### Relational analysis result of NS_A1_B1_B2_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9723904, upper bound: 380.9750522
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_B2_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -111.4611664, 285.2607117, -121.7696304, 310.1657410, -421.6268921, 407.0302429
1: -282.1170044, 431.5863953, -306.4219055, 468.5155945, -750.6325684, 738.0082397
2: -182.2482147, 421.4548950, -200.0270386, 458.1473083, -640.3955078, 621.4819336
3: -304.7623901, 498.6835632, -330.1451416, 542.2260742, -846.9884644, 828.8286133
4: -265.4147644, 481.4683838, -290.7833557, 523.0679932, -788.4827271, 772.2517090

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B2_A2_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_B2_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9734531, upper bound: 380.9742195
time: 0.69 seconds

## Relational analysis of NS_A1_B1_B2_A2_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_B2_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9734531, upper bound: 380.9754586
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_B2_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -102.6811066, 263.8839111, -106.1054459, 268.0679932, -370.7490845, 369.9893188
1: -260.1919861, 399.8999939, -267.1788940, 403.8349304, -664.0267334, 667.0788574
2: -167.7418671, 389.2119446, -174.6351471, 394.5845032, -562.3261108, 563.8470459
3: -281.0827026, 461.7192688, -288.0542603, 466.9308472, -748.0134277, 749.7735596
4: -244.5398712, 445.4766235, -252.9153900, 450.5075378, -695.0471191, 698.3919678

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_B2_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_B2_A2_B2_B1_B1_B1

### Relational analysis result of NS_A1_B1_B2_A2_B2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -380.9589829, upper bound: 380.9603389
time: 0.75 seconds

## Relational analysis of NS_A1_B1_B2_A2_B2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_B2_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_B2_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_B2_A2_B2_B1_B1_B1

### Relational analysis result of NS_A1_B1_B2_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9656173, upper bound: 380.9655929
time: 1.19 seconds

## Relational analysis of NS_A1_B1_B2_A2_B2_B1_B1_B2

### Relational analysis result of NS_A1_B1_B2_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9651927, upper bound: 380.9650283
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_B2_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -107.9074936, 276.5903320, -122.3802032, 310.8665771, -418.7740784, 398.9705200
1: -273.1858215, 418.6239624, -307.7767639, 468.8091125, -741.9949341, 726.4006958
2: -176.2169189, 408.6260071, -201.4125366, 458.4449768, -634.6618652, 610.0385742
3: -295.2728271, 483.6995544, -332.4176636, 542.3341064, -837.6068115, 816.1171875
4: -256.6654053, 466.9081421, -291.9450073, 523.3766479, -780.0419922, 758.8531494

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_B2_A2_B2_B1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9677586, upper bound: 380.9690672
time: 0.65 seconds

## Relational analysis of NS_A1_B1_B2_A2_B2_B1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9675457, upper bound: 380.9692741
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_B2_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -108.9098129, 278.9192810, -122.9622726, 311.3305969, -420.2403870, 401.8815308
1: -275.5817871, 422.0393372, -308.4754944, 469.6871033, -745.2689209, 730.5148315
2: -178.0475769, 412.0944214, -202.3161621, 459.4302979, -637.4778442, 614.4105225
3: -297.7454834, 487.6892090, -333.1696472, 543.4403076, -841.1857300, 820.8587036
4: -259.2872009, 470.8419800, -293.5801086, 524.0889893, -783.3762207, 764.4220581

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_B2_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_B2_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B2_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B2_A2_B2_B2_B1_B1

### Relational analysis result of NS_A1_B1_B2_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9720970, upper bound: 380.9723984
time: 0.74 seconds

## Relational analysis of NS_A1_B1_B2_A2_B2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_B2_A2_B2_B2_B1_B1

### Relational analysis result of NS_A1_B1_B2_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9724123, upper bound: 380.9718516
time: 0.65 seconds

## Relational analysis of NS_A1_B1_B2_A2_B2_B2_B1_B2

### Relational analysis result of NS_A1_B1_B2_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9720324, upper bound: 380.9707180
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_B2_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -111.2489700, 284.7507324, -126.0924911, 320.0744629, -431.3234253, 410.8432007
1: -281.6785583, 430.7590027, -316.6499329, 483.0407410, -764.7191772, 747.4089355
2: -181.9226837, 420.7587891, -207.2400818, 472.0992737, -654.0219727, 627.9989014
3: -304.1784058, 497.8001099, -342.0503845, 558.7131348, -862.8914185, 839.8504639
4: -264.7868958, 480.6239014, -301.0981140, 538.7625732, -803.5494385, 781.7218628

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_B2_A2_B2_B2_B2_A1

### Relational analysis result of NS_A1_B1_B2_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9762282, upper bound: 380.9768545
time: 0.94 seconds

## Relational analysis of NS_A1_B1_B2_A2_B2_B2_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -380.9762282, upper bound: 380.9768545
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -115.2364197, 294.7889099, -109.4662857, 280.2384338, -395.4748535, 404.2551575
1: -291.6886902, 445.7273865, -278.6857300, 423.9806824, -715.6693726, 724.4130249
2: -189.3491364, 436.9789429, -179.9723053, 414.4336548, -603.7827759, 616.9512329
3: -313.8222046, 516.0974731, -299.3610840, 490.8939514, -804.7161865, 815.4585571
4: -274.4563293, 498.2303772, -260.1336975, 474.2632751, -748.7196045, 758.3640747

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_A2_B1_A1_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -115.5150986, 295.5003967, -109.4662857, 280.2384338, -395.7534790, 404.9666443
1: -292.4719238, 446.8037109, -278.6857300, 423.9806824, -716.4526367, 725.4893188
2: -189.8270264, 438.0471802, -179.9723053, 414.4336548, -604.2606812, 618.0193481
3: -314.6297607, 517.3646240, -299.3610840, 490.8939514, -805.5236816, 816.7257080
4: -275.0808716, 499.4831238, -260.1336975, 474.2632751, -749.3441162, 759.6167603

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_A2_B1_A1_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -144.7732697, 368.1312256, -109.4662857, 280.2384338, -425.0116882, 477.5975037
1: -364.0415344, 554.6885376, -278.6857300, 423.9806824, -788.0222168, 833.3742676
2: -238.2898865, 544.7761230, -179.9723053, 414.4336548, -652.7235107, 724.7484131
3: -391.9978333, 643.0028076, -299.3610840, 490.8939514, -882.8916626, 942.3638916
4: -344.8063660, 620.4104004, -260.1336975, 474.2632751, -819.0695801, 880.5440674

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 34

## BFS NS instance: NS_A2_B1_A1_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -145.0695496, 368.8973999, -109.4662857, 280.2384338, -425.3079224, 478.3636780
1: -364.8708496, 555.8542480, -278.6857300, 423.9806824, -788.8515015, 834.5399170
2: -238.7968292, 545.9258423, -179.9723053, 414.4336548, -653.2304688, 725.8981323
3: -392.8538208, 644.3648682, -299.3610840, 490.8939514, -883.7478027, 943.7259521
4: -345.4872131, 621.7465820, -260.1336975, 474.2632751, -819.7504883, 881.8802490

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 34

## BFS NS instance: NS_A2_B1_A1_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -115.2364197, 294.7889099, -109.7886734, 281.0440674, -396.2804871, 404.5775452
1: -291.6886902, 445.7273865, -279.5932922, 425.1916199, -716.8803101, 725.3206787
2: -189.3491364, 436.9789429, -180.5375671, 415.6315613, -604.9807129, 617.5164795
3: -313.8222046, 516.0974731, -300.2929993, 492.3227844, -806.1450195, 816.3903198
4: -274.4563293, 498.2303772, -260.8638306, 475.6730042, -750.1293335, 759.0941772

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_A2_B1_A1_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -115.5150986, 295.5003967, -109.7886734, 281.0440674, -396.5591125, 405.2890625
1: -292.4719238, 446.8037109, -279.5932922, 425.1916199, -717.6635132, 726.3969727
2: -189.8270264, 438.0471802, -180.5375671, 415.6315613, -605.4586182, 618.5846558
3: -314.6297607, 517.3646240, -300.2929993, 492.3227844, -806.9525146, 817.6575928
4: -275.0808716, 499.4831238, -260.8638306, 475.6730042, -750.7539062, 760.3469238

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_A2_B1_A1_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -144.7732697, 368.1312256, -109.7886734, 281.0440674, -425.8172913, 477.9198608
1: -364.0415344, 554.6885376, -279.5932922, 425.1916199, -789.2331543, 834.2818604
2: -238.2898865, 544.7761230, -180.5375671, 415.6315613, -653.9213867, 725.3135986
3: -391.9978333, 643.0028076, -300.2929993, 492.3227844, -884.3206177, 943.2957153
4: -344.8063660, 620.4104004, -260.8638306, 475.6730042, -820.4793701, 881.2742310

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 34

## BFS NS instance: NS_A2_B1_A1_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -145.0695496, 368.8973999, -109.7886734, 281.0440674, -426.1135559, 478.6860352
1: -364.8708496, 555.8542480, -279.5932922, 425.1916199, -790.0622559, 835.4475098
2: -238.7968292, 545.9258423, -180.5375671, 415.6315613, -654.4283447, 726.4633789
3: -392.8538208, 644.3648682, -300.2929993, 492.3227844, -885.1766357, 944.6577759
4: -345.4872131, 621.7465820, -260.8638306, 475.6730042, -821.1602173, 882.6104126

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 34

## BFS NS instance: NS_A2_B1_A1_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -115.2364197, 294.7889099, -111.4585800, 285.5787048, -400.8151245, 406.2474670
1: -291.6886902, 445.7273865, -281.9114075, 432.1082764, -723.7969971, 727.6386719
2: -189.3491364, 436.9789429, -182.2229309, 422.0971375, -611.4462891, 619.2019043
3: -313.8222046, 516.0974731, -304.4627991, 499.5685120, -813.3906250, 820.5601807
4: -274.4563293, 498.2303772, -265.2369995, 482.2981567, -756.7545166, 763.4674072

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 34

## BFS NS instance: NS_A2_B1_A1_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -115.5150986, 295.5003967, -111.4585800, 285.5787048, -401.0937195, 406.9589844
1: -292.4719238, 446.8037109, -281.9114075, 432.1082764, -724.5802002, 728.7150269
2: -189.8270264, 438.0471802, -182.2229309, 422.0971375, -611.9241943, 620.2700806
3: -314.6297607, 517.3646240, -304.4627991, 499.5685120, -814.1982422, 821.8273926
4: -275.0808716, 499.4831238, -265.2369995, 482.2981567, -757.3790283, 764.7200928

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 34

## BFS NS instance: NS_A2_B1_A1_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -144.7732697, 368.1312256, -111.4585800, 285.5787048, -430.3518982, 479.5898132
1: -364.0415344, 554.6885376, -281.9114075, 432.1082764, -796.1497803, 836.5999756
2: -238.2898865, 544.7761230, -182.2229309, 422.0971375, -660.3870239, 726.9990234
3: -391.9978333, 643.0028076, -304.4627991, 499.5685120, -891.5661621, 947.4655762
4: -344.8063660, 620.4104004, -265.2369995, 482.2981567, -827.1044922, 885.6473999

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 34

## BFS NS instance: NS_A2_B1_A1_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -145.0695496, 368.8973999, -111.4585800, 285.5787048, -430.6481628, 480.3559875
1: -364.8708496, 555.8542480, -281.9114075, 432.1082764, -796.9790649, 837.7656250
2: -238.7968292, 545.9258423, -182.2229309, 422.0971375, -660.8939819, 728.1488037
3: -392.8538208, 644.3648682, -304.4627991, 499.5685120, -892.4223022, 948.8276367
4: -345.4872131, 621.7465820, -265.2369995, 482.2981567, -827.7854004, 886.9835815

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 34

## BFS NS instance: NS_A2_B1_A1_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -115.2364197, 294.7889099, -111.7214584, 286.2406921, -401.4771118, 406.5103455
1: -291.6886902, 445.7273865, -282.6607971, 433.1039124, -724.7926025, 728.3881836
2: -189.3491364, 436.9789429, -182.6893921, 423.0865479, -612.4356689, 619.6683350
3: -313.8222046, 516.0974731, -305.2257690, 500.7460022, -814.5682373, 821.3229370
4: -274.4563293, 498.2303772, -265.8354797, 483.4584351, -757.9147949, 764.0658569

Time for backsubstitution: 0.90 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.70 + 417.62 = 420.32 seconds
